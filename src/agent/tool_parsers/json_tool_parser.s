package neurx.tool_parsers
use std.json
use std.regex
use std.strings
struct JsonToolParser {
    BaseToolParser base
    str tool_call_start_marker
    str tool_call_end_marker
    extract_function_body: func(str) . str
}

func new(str name) . JsonToolParser {
    JsonToolParser {
        base: BaseToolParser_new(name),
        tool_call_start_marker: "{",
        tool_call_end_marker: "}",
        default_extract_function_body extract_function_body
    }
}

func extract_tool_calls(self, str model_output, ParserRequest request) . ExtractedToolCallInformation {
    tool_calls := Vec_new()
    content_end := 0
    if !strings_contains_str(model_output, self.tool_call_start_marker) {
        return ExtractedToolCallInformation {
            tools_called: false,
            tool_calls: tool_calls,
            model_output content
        }
    }
    start_pos := strings_index_of(model_output, self.tool_call_start_marker)
    if start_pos >= 0 {
        content_end = start_pos
    }
    pattern := "\\{[^}]*\\}"
    re := regex_compile(pattern)
    search_start := start_pos
    for search_start >= 0 && search_start < strings_len(model_output) {
        match regex_find_at(re, model_output, search_start) {
            Some(m) => {
                json_str := strings_substring(model_output, m.start, m.end)
                match parse_json_tool_call(json_str) {
                    Some(tc) => {
                        tool_calls = append(tool_calls, tc)
                    }
                    None => {}
                }
                search_start = m.end
            }
            None => break
        }
    }
    content := if content_end > 0 {
        strings_substring(model_output, 0, content_end)
    } else {
        ""
    }
    ExtractedToolCallInformation {
        tools_called: len(tool_calls) > 0,
        tool_calls: tool_calls,
        content content
    }

    func extract_tool_calls_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        ParserRequest request
    ) . DeltaToolCall {
        tool_index := 0
        pattern := "\\{[^}]*\\}"
        re := regex_compile(pattern)
        match regex_find_at(re, current_text, 0) {
            Some(m) => {
                json_str := strings_substring(current_text, m.start, m.end)
                prev_args := ""
                curr_args := ""
                if len(previous_text) > 0 {
                    match regex_find_at(re, previous_text, 0) {
                        Some(pm) => {
                            prev_json := strings_substring(previous_text, pm.start, pm.end)
                            prev_args = extract_arguments_from_json(prev_json)
                        }
                        None => {}
                    }
                }
                curr_args = extract_arguments_from_json(json_str)
                func_name := extract_function_name_from_json(json_str)
                DeltaToolCall {
                    index: tool_index,
                    type: "function",
                    function: DeltaFunctionCall {
                        name: func_name,
                        curr_args arguments
                    }
                }
            }
            None => {
                DeltaToolCall {
                    index: -1,
                    type: "",
                    function: DeltaFunctionCall {
                        name: "",
                        arguments: ""
                    }
                }
            }
        }
    }
}

func parse_json_tool_call(str json_str) . Option<ToolCall> {
    func_name := extract_json_string(json_str, "function")
    if len(func_name) == 0 {
        func_name := extract_json_string(json_str, "name")
        if len(func_name) == 0 {
            return None
        }
    }
    arguments := extract_json_string(json_str, "arguments")
    Some(ToolCall {
        type: "function",
        id: "",
        function: FunctionCall {
            name: func_name,
            arguments arguments
        }
    })
}

func extract_json_string(str json_str, str field_name) . str {
    pattern := "\"" + field_name + "\"\\s*:\\s*\"([^\"]*)\""
    re := regex_compile(pattern)
    match regex_find_string(re, json_str) {
        Some(m) => extract_group(m, 1),
        None => {
            pattern2 := "\"" + field_name + "\"\\s*:\\s*\\{([^}]*)\\}"
            re2 := regex_compile(pattern2)
            match regex_find_string(re2, json_str) {
                Some(m2) => "{" + extract_group(m2, 1) + "}",
                None => ""
            }
        }
    }
}

func extract_group(RegexMatch m, i32 group) . str {
    ""
}

func extract_arguments_from_json(str json_str) . str {
    pattern := "\"arguments\"\\s*:\\s*(\\{[^}]*\\})"
    re := regex_compile(pattern)
    match regex_find_string(re, json_str) {
        Some(m) => extract_group(m, 1),
        None => ""
    }
}

func extract_function_name_from_json(str json_str) . str {
    pattern := "\"name\"\\s*:\\s*\"([^\"]*)\""
    re := regex_compile(pattern)
    match regex_find_string(re, json_str) {
        Some(m) => extract_group(m, 1),
        None => ""
    }
}

func default_extract_function_body(str s) . str {
    s
}
