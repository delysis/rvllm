#import <Foundation/Foundation.h>
#import <CoreML/CoreML.h>
#import <objc/runtime.h>
#include <dlfcn.h>

int main() {
    void *handle = dlopen("/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine", RTLD_LAZY);
    if (!handle) {
        printf("Failed to load ANE\n");
        return 1;
    }
    
    NSURL *url = [NSURL fileURLWithPath:@"/tmp/rvllm_debug_workspace/model.mlmodelc"];
    NSError *error = nil;
    MLModel *model = [MLModel modelWithContentsOfURL:url error:&error];
    if (model) {
        printf("Model loaded successfully!\n");
        id engine = [model performSelector:NSSelectorFromString(@"internalEngine")];
        printf("Internal Engine: %s\n", [[engine description] UTF8String]);
        
        // Inspect engine class
        Class engineClass = [engine class];
        printf("Engine Class: %s\n", [NSStringFromClass(engineClass) UTF8String]);
    } else {
        printf("Model load failed: %s\n", [[error localizedDescription] UTF8String]);
    }
    
    return 0;
}
