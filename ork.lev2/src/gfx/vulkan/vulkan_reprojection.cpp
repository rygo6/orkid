////////////////////////////////////////////////////////////////
// Orkid Media Engine
// Copyright 1996-2023, Michael T. Mayers.
// Distributed under the MIT License.
// see license-mit.txt in the root of the repo, and/or https://opensource.org/license/mit/
////////////////////////////////////////////////////////////////

#include "headers/vulkan_ctx.h"

///////////////////////////////////////////////////////////////////////////////
namespace ork::lev2::vulkan {
///////////////////////////////////////////////////////////////////////////////
static auto logchan_reproj = logger()->configureChannel("VKREPROJ", fvec3(0.2, 0.8, 0.6), true);

///////////////////////////////////////////////////////////////////////////////

VkReprojectionContext::VkReprojectionContext(vkcontext_rawptr_t ctxVK, VkOutputMode mode)
    : _contextVK(ctxVK)
    , _mode(mode) {
}

///////////////////////////////////////////////////////////////////////////////

VkReprojectionContext::~VkReprojectionContext() {
  _teardown();
}

///////////////////////////////////////////////////////////////////////////////

void VkReprojectionContext::_buildup() {
  OrkAssert(_contextVK != nullptr);
  _gfxRenderComplete = std::make_shared<VulkanBinarySemaphore>(_contextVK);
  logchan_reproj->log("VkReprojectionContext::_buildup mode=%d", (int)_mode);
}

///////////////////////////////////////////////////////////////////////////////

void VkReprojectionContext::_teardown() {
  _gfxRenderComplete = nullptr;
  _frameFences.clear();
  _swapchain = nullptr;
#if defined(__linux__)
  _swapchain_drm = nullptr;
#endif
  logchan_reproj->log("VkReprojectionContext::_teardown");
}

///////////////////////////////////////////////////////////////////////////////

void VkReprojectionContext::submitAndPresent(vkcontext_rawptr_t ctxVK) {
  OrkAssert(_mode == VkOutputMode::GFX_REPROJECT_SWAP);
  OrkAssert(_swapchain != nullptr);
  OrkAssert(_gfxRenderComplete != nullptr);

  // 1. Acquire swapchain image (handles resize / reinit internally)
  _swapchain->_update();
  size_t sub_index = _swapchain->subIndex();

  // 2. Get source image: main RTG offscreen color buffer 0
  auto main_rtg = ctxVK->_fbi->_ensureMainRtg();
  OrkAssert(main_rtg != nullptr);
  auto src_rtb  = main_rtg->buffer(0);
  OrkAssert(src_rtb != nullptr);
  auto src_impl = src_rtb->_impl.getShared<VklRtBufferImpl>();
  OrkAssert(src_impl != nullptr);
  OrkAssert(src_impl->_imgobj != nullptr);

  VkImage src_image  = src_impl->_imgobj->_vkimage;
  int     src_w      = main_rtg->miW;
  int     src_h      = main_rtg->miH;
  VkImageLayout src_layout = src_impl->_currentLayout;

  // 3. Get destination image: acquired swapchain image
  OrkAssert(_swapchain->_curSwapWriteImage != 0xffffffff);
  auto dst_imgobj = _swapchain->_swapChainImages[_swapchain->_curSwapWriteImage];
  OrkAssert(dst_imgobj != nullptr);
  VkImage dst_image = dst_imgobj->_vkimage;
  int     dst_w     = _swapchain->_width;
  int     dst_h     = _swapchain->_height;

  logchan_reproj->log("submitAndPresent: src=%dx%d -> dst=%dx%d sub_index=%zu", src_w, src_h, dst_w, dst_h, sub_index);

  // 4. Allocate a one-shot primary command buffer for the blit
  VkCommandBufferAllocateInfo allocInfo{
    VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
    .commandPool        = ctxVK->_vkcmdpool_graphics,
    .level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
    .commandBufferCount = 1,
  };

  VkCommandBuffer blitCB = VK_NULL_HANDLE;
  vkAllocateCommandBuffers(ctxVK->_vkdevice, &allocInfo, &blitCB);
  OrkAssert(blitCB != VK_NULL_HANDLE);

  VkCommandBufferBeginInfo beginInfo{
    VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
    .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
  };
  vkBeginCommandBuffer(blitCB, &beginInfo);

  // 5. Transition source: current layout → TRANSFER_SRC_OPTIMAL
  auto src_barrier = createImageBarrier(
      src_image,
      src_layout,
      VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
      VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
      VK_ACCESS_TRANSFER_READ_BIT);
  src_barrier->subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;

  vkCmdPipelineBarrier(
      blitCB,
      VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
      VK_PIPELINE_STAGE_TRANSFER_BIT,
      0, 0, nullptr, 0, nullptr,
      1, src_barrier.get());

  // 6. Transition destination: UNDEFINED → TRANSFER_DST_OPTIMAL
  //    (UNDEFINED discards old contents — correct since we're fully overwriting)
  auto dst_barrier = createImageBarrier(
      dst_image,
      VK_IMAGE_LAYOUT_UNDEFINED,
      VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
      VkAccessFlagBits(0),
      VK_ACCESS_TRANSFER_WRITE_BIT);
  dst_barrier->subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;

  vkCmdPipelineBarrier(
      blitCB,
      VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
      VK_PIPELINE_STAGE_TRANSFER_BIT,
      0, 0, nullptr, 0, nullptr,
      1, dst_barrier.get());

  // 7. Blit source → destination (pass-through; future: warp using matrices)
  VkImageBlit blit{
    .srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
    .srcOffsets     = {{0, 0, 0}, {src_w, src_h, 1}},
    .dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
    .dstOffsets     = {{0, 0, 0}, {dst_w, dst_h, 1}},
  };

  vkCmdBlitImage(
      blitCB,
      src_image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
      dst_image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
      1, &blit,
      VK_FILTER_LINEAR);

  // 8. Transition destination: TRANSFER_DST_OPTIMAL → PRESENT_SRC_KHR
  auto dst_barrier2 = createImageBarrier(
      dst_image,
      VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
      VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
      VK_ACCESS_TRANSFER_WRITE_BIT,
      VkAccessFlagBits(0));
  dst_barrier2->subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;

  vkCmdPipelineBarrier(
      blitCB,
      VK_PIPELINE_STAGE_TRANSFER_BIT,
      VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
      0, 0, nullptr, 0, nullptr,
      1, dst_barrier2.get());

  // 9. Transition source back to COLOR_ATTACHMENT_OPTIMAL for next frame
  auto src_barrier2 = createImageBarrier(
      src_image,
      VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
      VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
      VK_ACCESS_TRANSFER_READ_BIT,
      VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT);
  src_barrier2->subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;

  vkCmdPipelineBarrier(
      blitCB,
      VK_PIPELINE_STAGE_TRANSFER_BIT,
      VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
      0, 0, nullptr, 0, nullptr,
      1, src_barrier2.get());

  vkEndCommandBuffer(blitCB);

  // Update tracked layout
  src_impl->_currentLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
  if (src_impl->_imgobj)
    src_impl->_imgobj->_currentLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

  // 10. Submit blit CB:
  //     wait on imageAcquiredSemaphores[sub_index] + gfxRenderComplete
  //     signal renderCompleteSemaphores[sub_index]
  //     fence = swapchain's frameFences[sub_index]
  VkSemaphore wait_semas[2] = {
      _swapchain->_imageAcquiredSemaphores[sub_index]->_vksema,
      _gfxRenderComplete->_vksema,
  };
  VkPipelineStageFlags wait_stages[2] = {
      VK_PIPELINE_STAGE_TRANSFER_BIT,
      VK_PIPELINE_STAGE_TRANSFER_BIT,
  };

  VkSubmitInfo SI{
    VK_STRUCTURE_TYPE_SUBMIT_INFO,
    .waitSemaphoreCount   = 2,
    .pWaitSemaphores      = wait_semas,
    .pWaitDstStageMask    = wait_stages,
    .commandBufferCount   = 1,
    .pCommandBuffers      = &blitCB,
    .signalSemaphoreCount = 1,
    .pSignalSemaphores    = &_swapchain->_renderCompleteSemaphores[sub_index]->_vksema,
  };

  auto& fence = _swapchain->_frameFences[sub_index];
  fence->reset();
  vkQueueSubmit(ctxVK->_vkqueue_graphics, 1, &SI, fence->_vkfence);

  // 11. Present + wait
  _swapchain->enqueuePresentFrame(ctxVK);
  _swapchain->waitPresentFrame(ctxVK);

  // 12. Free one-shot command buffer
  vkFreeCommandBuffers(ctxVK->_vkdevice, ctxVK->_vkcmdpool_graphics, 1, &blitCB);

  _currentFrame++;
}

///////////////////////////////////////////////////////////////////////////////

void VkReprojectionContext::setReprojectionMatrices(
    const fmtx4& prev_view,
    const fmtx4& curr_view,
    const fmtx4& proj) {
  std::lock_guard<std::mutex> lock(_matrixMutex);
  _prevViewMatrix = prev_view;
  _currViewMatrix = curr_view;
  _projMatrix     = proj;
}

///////////////////////////////////////////////////////////////////////////////
} // namespace ork::lev2::vulkan
///////////////////////////////////////////////////////////////////////////////
