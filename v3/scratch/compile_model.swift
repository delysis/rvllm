import CoreML
import Foundation

let args = ProcessInfo.processInfo.arguments
if args.count < 3 {
    print("Usage: compile <input.mlmodel> <output_dir>")
    exit(1)
}

let inputURL = URL(fileURLWithPath: args[1])
let outputURL = URL(fileURLWithPath: args[2])

do {
    let compiledURL = try MLModel.compileModel(at: inputURL)
    print("Compiled to: \(compiledURL.path)")
    
    // Copy to output_dir
    let fileManager = FileManager.default
    if fileManager.fileExists(atPath: outputURL.path) {
        try fileManager.removeItem(at: outputURL)
    }
    try fileManager.copyItem(at: compiledURL, to: outputURL)
} catch {
    print("Error: \(error)")
    exit(1)
}
