/*
 * metalq.m - Metal-Q API Implementation
 */

#import "metalq.h"
#import "context_internal.h"
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <dlfcn.h>

// ===========================================================================
// Context Management
// ===========================================================================

bool metalq_is_supported(void) {
  @autoreleasepool {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    return (device != nil);
  }
}

mq_context_t metalq_create_context(void) {
  @autoreleasepool {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (!device)
      return NULL;

    id<MTLCommandQueue> queue = [device newCommandQueue];
    if (!queue)
      return NULL;

    NSError *error = nil;
    id<MTLLibrary> library =
        [device newDefaultLibraryWithBundle:[NSBundle mainBundle] error:&error];

    // In scripts, mainBundle might not have .metallib. Check adjacent file.
    if (!library) {
      NSMutableArray *searchPaths = [NSMutableArray array];

      // Resolve the directory this dylib itself was loaded from via dladdr on
      // one of our own exported symbols. The build places libmetalq.dylib and
      // default.metallib side by side in native/build/, so this normally
      // finds the library regardless of the caller's CWD. Tried first, before
      // the CWD-relative fallbacks below.
      Dl_info info;
      if (dladdr((void *)metalq_create_context, &info) && info.dli_fname) {
        NSString *dylibPath = [NSString stringWithUTF8String:info.dli_fname];
        NSString *dylibDir = [dylibPath stringByDeletingLastPathComponent];
        [searchPaths
            addObject:[dylibDir stringByAppendingPathComponent:@"default.metallib"]];
      }

      [searchPaths addObjectsFromArray:@[
        @"./default.metallib",                  // Current dir
        @"native/build/default.metallib",       // From project root
        @"../native/build/default.metallib",    // From examples/
        @"../../native/build/default.metallib", // From deep nesting
        @"build/default.metallib",              // Direct build
        [NSString stringWithFormat:@"%@/native/build/default.metallib",
                                   [[[NSFileManager defaultManager]
                                       currentDirectoryPath]
                                       stringByDeletingLastPathComponent]],
        [NSString stringWithFormat:@"%@/build/default.metallib",
                                   [[NSBundle mainBundle] resourcePath]]
      ]];

      for (NSString *path in searchPaths) {
        if ([[NSFileManager defaultManager] fileExistsAtPath:path]) {
          library = [device newLibraryWithURL:[NSURL fileURLWithPath:path]
                                         error:&error];
          if (library) {
            NSLog(@"[MetalQ] Loaded Metal library from: %@", path);
            break;
          }
        }
      }
    }

    if (!library) {
      NSLog(@"[MetalQ] Warning: Failed to load library: %@", error);
    }

    NSMutableDictionary *pipelines = [NSMutableDictionary dictionary];
    NSMutableDictionary *bufferPool = [NSMutableDictionary dictionary];

    MetalQContext *ctx = (MetalQContext *)calloc(1, sizeof(MetalQContext));

    ctx->device = (__bridge_retained void *)device;
    ctx->commandQueue = (__bridge_retained void *)queue;
    ctx->library = library ? (__bridge_retained void *)library : NULL;
    ctx->pipelines = (__bridge_retained void *)pipelines;
    ctx->bufferPool = (__bridge_retained void *)bufferPool;

    return (mq_context_t)ctx;
  }
}

void metalq_destroy_context(mq_context_t ctx) {
  @autoreleasepool {
    if (!ctx)
      return;
    MetalQContext *mCtx = (MetalQContext *)ctx;

    if (mCtx->bufferPool) {
      CFRelease(mCtx->bufferPool);
      mCtx->bufferPool = NULL;
    }
    if (mCtx->pipelines) {
      CFRelease(mCtx->pipelines);
      mCtx->pipelines = NULL;
    }
    if (mCtx->library) {
      CFRelease(mCtx->library);
      mCtx->library = NULL;
    }
    if (mCtx->commandQueue) {
      CFRelease(mCtx->commandQueue);
      mCtx->commandQueue = NULL;
    }
    if (mCtx->device) {
      CFRelease(mCtx->device);
      mCtx->device = NULL;
    }

    free(mCtx);
  }
}
