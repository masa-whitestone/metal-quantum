/*
 * metalq.m - Metal-Q API Implementation
 */

#import "metalq.h"
#import "context_internal.h"
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

// ===========================================================================
// Context Management
// ===========================================================================

bool metalq_is_supported(void) {
  id<MTLDevice> device = MTLCreateSystemDefaultDevice();
  return (device != nil);
}

mq_context_t metalq_create_context(void) {
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
    NSArray *searchPaths = @[
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
    ];

    for (NSString *path in searchPaths) {
      if ([[NSFileManager defaultManager] fileExistsAtPath:path]) {
        library = [device newLibraryWithFile:path error:&error];
        if (library) {
          break;
        }
      }
    }
  }

  if (!library) {
    NSLog(@"[MetalQ] Warning: Failed to load library: %@", error);
  }

  NSMutableDictionary *pipelines = [NSMutableDictionary dictionary];

  MetalQContext *ctx = (MetalQContext *)calloc(1, sizeof(MetalQContext));

  ctx->device = (__bridge_retained void *)device;
  ctx->commandQueue = (__bridge_retained void *)queue;
  ctx->library = library ? (__bridge_retained void *)library : NULL;
  ctx->pipelines = (__bridge_retained void *)pipelines;

  return (mq_context_t)ctx;
}

void metalq_destroy_context(mq_context_t ctx) {
  if (!ctx)
    return;
  MetalQContext *mCtx = (MetalQContext *)ctx;

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
