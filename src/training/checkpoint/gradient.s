package neurx.checkpoint.gradient
import "neurx.autograd"
struct checkpoint_config {
    bool enabled
    int checkpoints_per_layer
    bool preserve_inputs
    int recomputation_strategy
}

struct checkpoint_state {
    saved_tensors: [][]autograd.tensor
    int recomputation_count
    float memory_saved
}

struct checkpoint_layer {
    layer_fn: func([][]autograd.tensor) [][]autograd.tensor
    inputs: [][]autograd.tensor
    outputs: [][]autograd.tensor
    bool needs_recompute
}

func new_checkpoint_config(bool enabled) checkpoint_config {
    checkpoint_config config {
        enabled: enabled,
        checkpoints_per_layer: 1,
        preserve_inputs: true,
        recomputation_strategy: 0,
    }
    config
}

func new_checkpoint_state() checkpoint_state {
    checkpoint_state state {
        saved_tensors: [][]autograd.tensor{},
        recomputation_count: 0,
        memory_saved: 0.0,
    }
    state
}

func checkpoint_wrapper(func layer_fn, []autograd.tensor inputs, checkpoint_config config) []autograd.tensor {
    if !config.enabled {
        return layer_fn(inputs...)
    }
    autograd.disable_grad()
    []autograd.tensor outputs = layer_fn(inputs...)
    autograd.enable_grad()
    for i := 0; i < len(inputs); i += 1 {
        autograd.retain_grad(inputs[i])
    }
    []autograd.tensor detached_outputs = make([]autograd.tensor, len(outputs))
    for i := 0; i < len(outputs); i += 1 {
        detached_outputs = append(detached_outputs, autograd.tensor_detach(outputs[i]))
    }

    func backward_fn([]autograd.tensor grads) []autograd.tensor {
        autograd.enable_grad()
        []autograd.tensor recomputed_outputs = layer_fn(inputs...)
        []autograd.tensor result_grads = make([]autograd.tensor, len(inputs))
        for i := 0; i < len(recomputed_outputs); i += 1 {
            autograd.backward_with_grad(recomputed_outputs[i], grads[i])
        }
        for i := 0; i < len(inputs); i += 1 {
            result_grads = append(result_grads, inputs[i].grad)
        }
        autograd.disable_grad()
        result_grads
    }
    for i := 0; i < len(detached_outputs); i += 1 {
        autograd.register_backward_hook(detached_outputs[i], backward_fn)
    }
    detached_outputs
}

func checkpoint_module(pointer module, []autograd.tensor inputs, checkpoint_config config) []autograd.tensor {
    if !config.enabled {
        return module.forward(inputs...)
    }
    autograd.disable_grad()
    []autograd.tensor outputs = module.forward(inputs...)
    autograd.enable_grad()
    for i := 0; i < len(inputs); i += 1 {
        autograd.retain_grad(inputs[i])
    }
    []autograd.tensor detached_outputs = make([]autograd.tensor, len(outputs))
    for i := 0; i < len(outputs); i += 1 {
        detached_outputs = append(detached_outputs, autograd.tensor_detach(outputs[i]))
    }

    func recompute_fn() []autograd.tensor {
        autograd.enable_grad()
        []autograd.tensor result = module.forward(inputs...)
        autograd.disable_grad()
        result
    }
    for i := 0; i < len(detached_outputs); i += 1 {
        autograd.register_backward_hook(detached_outputs[i], func(grads) []autograd.tensor {
            []autograd.tensor recomputed = recompute_fn()
            []autograd.tensor input_grads = make([]autograd.tensor, len(inputs))
            for j := 0; j < len(recomputed); j += 1 {
                autograd.backward_with_grad(recomputed[j], grads[j])
            }
            for j := 0; j < len(inputs); j += 1 {
                input_grads = append(input_grads, inputs[j].grad)
            }
            input_grads
        })
    }
    detached_outputs
}

func apply_checkpointing_to_transformer(pointer transformer, checkpoint_config config) pointer {
    if !config.enabled {
        return transformer
    }
    int num_layers = len(transformer.layers)
    for i := 0; i < num_layers; i += 1 {
        pointer layer = transformer.layers[i]
        func wrapped_forward([]autograd.tensor inputs) []autograd.tensor {
            checkpoint_wrapper(layer.forward, inputs, config)
        }
        layer.forward = wrapped_forward
    }
    transformer
}

func gradient_checkpointing_step(
    pointer model,
    []autograd.tensor inputs,
    func loss_fn,
    checkpoint_config config,
) float {
    autograd.zero_grad(model.parameters())
    []autograd.tensor outputs
    if config.enabled {
        outputs = apply_checkpointing_to_transformer(model, config).forward(inputs)
    } else {
        outputs = model.forward(inputs)
    }
    float loss = loss_fn(outputs)
    autograd.backward(loss)
    loss
}

func estimate_memory_savings(checkpoint_config config, int num_layers, int layer_memory_mb) float {
    if !config.enabled {
        return 0.0
    }
    float baseline = num_layers * layer_memory_mb
    float with_checkpointing = layer_memory_mb + (num_layers / config.checkpoints_per_layer) * layer_memory_mb
    baseline - with_checkpointing
}

func checkpoint_save_tensor(autograd.tensor tensor, checkpoint_state state) checkpoint_state {
    state.saved_tensors = append(state.saved_tensors, [tensor])
    state
}

func checkpoint_save_tensors([][]autograd.tensor tensors, checkpoint_state state) checkpoint_state {
    state.saved_tensors = append(state.saved_tensors, tensors)
    state
}

func checkpoint_clear(checkpoint_state state) checkpoint_state {
    state.saved_tensors = [][]autograd.tensor{}
    state.recomputation_count = 0
    state
}

func get_checkpoint_stats(checkpoint_state state) string {
    "checkpoint Stats: Recomputations=" + string(state.recomputation_count) +
    ", Memory Saved=" + string(state.memory_saved) + "MB"
}

func create_checkpoint_layer(func layer_fn) checkpoint_layer {
    checkpoint_layer layer {
        layer_fn: layer_fn,
        inputs: [][]autograd.tensor{},
        outputs: [][]autograd.tensor{},
        needs_recompute: false,
    }
    layer
}

func checkpoint_layer_forward(checkpoint_layer layer, []autograd.tensor inputs) []autograd.tensor {
    layer.inputs = [inputs]
    layer.outputs = [layer.layer_fn(inputs...)]
    layer.needs_recompute = true
    []autograd.tensor detached = make([]autograd.tensor, len(layer.outputs[0]))
    for i := 0; i < len(layer.outputs[0]); i += 1 {
        detached = append(detached, autograd.tensor_detach(layer.outputs[0][i]))
    }
    detached
}

func checkpoint_layer_backward(checkpoint_layer layer, []autograd.tensor grads) []autograd.tensor {
    if !layer.needs_recompute {
        return []autograd.tensor{}
    }
    autograd.enable_grad()
    []autograd.tensor recomputed = layer.layer_fn(layer.inputs[0]...)
    []autograd.tensor input_grads = make([]autograd.tensor, len(layer.inputs[0]))
    for i := 0; i < len(recomputed); i += 1 {
        autograd.backward_with_grad(recomputed[i], grads[i])
    }
    for i := 0; i < len(layer.inputs[0]); i += 1 {
        input_grads = append(input_grads, layer.inputs[0][i].grad)
    }
    autograd.disable_grad()
    layer.needs_recompute = false
    input_grads
}
