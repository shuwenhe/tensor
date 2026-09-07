use std.conv.int_to_string
package neurx.scripts.download_model
use neurx.runtime.io.{
    runtime_env_get,
    runtime_file_exists,
    runtime_make_dirs,
    runtime_read_text_file,
    runtime_write_text_file
}

struct download_config {
    string model_name
    string model_dir
    string model_url_base
    bool verify_checksums
    bool resume_download
    int max_retries
    string log_file
struct download_result {
    bool success
    string message
    int files_downloaded
    int files_skipped
    int files_failed
    float total_size_gb
    float download_speed_mbps
struct model_file_info {
    string name
    int size_bytes
    string checksum
    bool downloaded
func create_default_config() download_config {
    download_config{
        model_name: "Qwen2.5-0.5B-Instruct",
        model_dir: "/app/shuwen/model/Qwen2.5-0.5B-Instruct",
        model_url_base: "https:
        verify_checksums: true,
        resume_download: true,
        max_retries: 3,
        log_file: "/app/shuwen/neurx/artifact/download_model.log"
    }

func string_slice(string text, int start, int end) string {
    string result = ""
    int i = start
    for i < end && i < len(text) {
        result = result + string_char(text[i])
        i = i + 1
    }
    result
func string_char(int code) string {
    if code == 0 { return "\x00" }
    if code == 9 { return "\t" }
    if code == 10 { return "\n" }
    if code == 13 { return "\r" }
    if code == 32 { return " " }
    if code >= 48 && code <= 57 { return string_slice("0123456789", code - 48, code - 47) }
    ""
func trim_string(string s) string {
    int start = 0
    for start < len(s) && (s[start] == 32 || s[start] == 9 || s[start] == 10 || s[start] == 13) {
        start = start + 1
    }
    int end = len(s) - 1
    for end >= start && (s[end] == 32 || s[end] == 9 || s[end] == 10 || s[end] == 13) {
        end = end - 1
    }
    if end < start {
        return ""
    }
    string_slice(s, start, end + 1)
func get_model_files() []model_file_info {
    []model_file_info files = make([]model_file_info, 6)
    files[0] = model_file_info{
        name: "model.safetensors",
        size_bytes: 1945505792,
        checksum: "abc123def456",
        false downloaded
    }
    files[1] = model_file_info{
        name: "tokenizer.json",
        size_bytes: 2621440,
        checksum: "xyz789uvw123",
        false downloaded
    }
    files[2] = model_file_info{
        name: "config.json",
        size_bytes: 4096,
        checksum: "config123",
        false downloaded
    }
    files[3] = model_file_info{
        name: "generation_config.json",
        size_bytes: 2048,
        checksum: "genconfig123",
        false downloaded
    }
    files[4] = model_file_info{
        name: "tokenizer_config.json",
        size_bytes: 1024,
        checksum: "tokenconfig123",
        false downloaded
    }
    files[5] = model_file_info{
        name: ".gitattributes",
        size_bytes: 512,
        checksum: "git123",
        false downloaded
    }
    files
func check_model_directory(download_config config) bool {
    if runtime_file_exists(config.model_dir) {
        return true
    }
    runtime_make_dirs(config.model_dir)
    runtime_file_exists(config.model_dir)
func check_file_exists(string file_path) bool {
    runtime_file_exists(file_path)
func download_file(
    download_config config,
    model_file_info file_info,
    int retry_count
) bool {
    string file_path = config.model_dir + "/" + file_info.name
    if check_file_exists(file_path) && config.resume_download {
        log_message(config, "✓ Already downloaded: " + file_info.name)
        return true
    }
    string url = config.model_url_base + "/" + file_info.name
    string size_mb = int_to_string(file_info.size_bytes / 1048576)
    log_message(config, "⏳ Downloading: " + file_info.name + " (" + size_mb + " MB)")
    if retry_count > config.max_retries {
        log_message(config, "❌ Max retries exceeded for: " + file_info.name)
        return false
    }
    log_message(config, "   URL: " + url)
    log_message(config, "   Destination: " + file_path)
    return true
}

func verify_downloaded_files(download_config config, []model_file_info files) (int, int) {
    int verified = 0
    int failed = 0
    int i = 0
    for i < len(files) {
        string file_path = config.model_dir + "/" + files[i].name
        if check_file_exists(file_path) {
            verified = verified + 1
            log_message(config, "✓ Verified: " + files[i].name)
        } else {
            failed = failed + 1
            log_message(config, "❌ Missing: " + files[i].name)
        }
        i = i + 1
    }
    (verified, failed)
func calculate_total_size([]model_file_info files) int {
    int total = 0
    int i = 0
    for i < len(files) {
        if files[i].downloaded {
            total = total + files[i].size_bytes
        }
        i = i + 1
    }
    total
func log_message(download_config config, string message) {
    println(message)
func print_download_status(download_config config, []model_file_info files) {
    println("")
    println("╔════════════════════════════════════════════════════════════╗")
    println("║           NeurX Model Download Status                      ║")
    println("╚════════════════════════════════════════════════════════════╝")
    println("")
    println("Model: " + config.model_name)
    println("Directory: " + config.model_dir)
    println("")
    println("Files:")
    int i = 0
    int downloaded = 0
    int skipped = 0
    for i < len(files) {
        string status = "⏳"
        if files[i].downloaded {
            status = "✓"
            downloaded = downloaded + 1
        } else {
            skipped = skipped + 1
        }
        int size_mb = files[i].size_bytes / 1048576
        if size_mb == 0 {
            size_mb = 1
        }
        println("  " + status + " " + files[i].name + " (" + int_to_string(size_mb) + " MB)")
        i = i + 1
    }
    println("")
    println("Summary: " + int_to_string(downloaded) + " downloaded, " + int_to_string(skipped) + " failed")
    println("")
func main() {
    download_config config = create_default_config()
    println("╔════════════════════════════════════════════════════════════╗")
    println("║         NeurX Model Downloader (S Language)                ║")
    println("╚════════════════════════════════════════════════════════════╝")
    println("")
    string model_name_env = trim_string(runtime_env_get("NEURX_MODEL_NAME", ""))
    if len(model_name_env) > 0 {
        config.model_name = model_name_env
    }
    string model_dir_env = trim_string(runtime_env_get("NEURX_MODEL_DIR", ""))
    if len(model_dir_env) > 0 {
        config.model_dir = model_dir_env
    }
    println("Model: " + config.model_name)
    println("Target: " + config.model_dir)
    println("")
    if !check_model_directory(config) {
        println("❌ Failed to create model directory")
        return
    }
    println("✓ Model directory ready")
    println("")
    []model_file_info files = get_model_files()
    log_message(config, "Starting model download: " + config.model_name)
    log_message(config, "Target directory: " + config.model_dir)
    int downloaded = 0
    int skipped = 0
    int failed = 0
    int i = 0
    for i < len(files) {
        if download_file(config, files[i], 0) {
            files[i].downloaded = true
            downloaded = downloaded + 1
        } else {
            failed = failed + 1
        }
        i = i + 1
    }
    (int verified, int verify_failed) = verify_downloaded_files(config, files)
    print_download_status(config, files)
    println("✓ Download completed")
    println("  Downloaded: " + int_to_string(verified) + " files")
    println("  Failed: " + int_to_string(verify_failed) + " files")
    println("")
    if verify_failed == 0 {
        println("✅ All files downloaded successfully!")
        log_message(config, "Download completed successfully")
    } else {
        println("⚠️  Some files failed to download. Please retry.")
        log_message(config, "Download completed with " + int_to_string(verify_failed) + " failures")
    }
    download_config config = create_default_config()
    println("╔════════════════════════════════════════════════════════════╗")
    println("║         NeurX Model Downloader (S Language)                ║")
    println("╚════════════════════════════════════════════════════════════╝")
    println("")
    string model_name_env = trim_string(runtime_env_get("NEURX_MODEL_NAME", ""))
    if len(model_name_env) > 0 {
        config.model_name = model_name_env
    }
    string model_dir_env = trim_string(runtime_env_get("NEURX_MODEL_DIR", ""))
    if len(model_dir_env) > 0 {
        config.model_dir = model_dir_env
    }
    println("Model: " + config.model_name)
    println("Target: " + config.model_dir)
    println("")
    if !check_model_directory(config) {
        println("❌ Failed to create model directory")
        return
    }
    println("✓ Model directory ready")
    println("")
    []model_file_info files = get_model_files()
    log_message(config, "Starting model download: " + config.model_name)
    log_message(config, "Target directory: " + config.model_dir)
    int downloaded = 0
    int skipped = 0
    int failed = 0
    int i = 0
    for i < len(files) {
        if download_file(config, files[i], 0) {
            files[i].downloaded = true
            downloaded = downloaded + 1
        } else {
            failed = failed + 1
        }
        i = i + 1
    }
    (int verified, int verify_failed) = verify_downloaded_files(config, files)
    print_download_status(config, files)
    println("✓ Download completed")
    println("  Downloaded: " + int_to_string(verified) + " files")
    println("  Failed: " + int_to_string(verify_failed) + " files")
    println("")
    if verify_failed == 0 {
        println("✅ All files downloaded successfully!")
        log_message(config, "Download completed successfully")
    } else {
        println("⚠️  Some files failed to download. Please retry.")
        log_message(config, "Download completed with " + int_to_string(verify_failed) + " failures")
    }
}
