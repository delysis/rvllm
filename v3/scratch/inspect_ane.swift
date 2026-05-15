import Foundation
import AppleNeuralEngine

let client = _ANEClient.sharedConnection()
print("Client: \(client)")

// Inspect methods using reflection if possible, or just try to call it.
// We can use Objective-C runtime to find the signature.
let method = class_getInstanceMethod(_ANEClient.self, Selector(("compileModel:options:qos:error:")))
if let method = method {
    let encoding = method_getTypeEncoding(method)
    if let encoding = encoding {
        print("Encoding: \(String(cString: encoding))")
    }
}
