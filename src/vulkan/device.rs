//! Vulkan device, memory allocator, buffer upload/download, and command submission.

use crate::{Error, Result};
use ash::{Entry, vk};
use std::{ffi::CStr, sync::Arc};

pub(super) fn failure(message: impl std::fmt::Display) -> Error {
    Error::Backend(format!("Vulkan: {message}"))
}

pub(super) struct VkInner {
    pub(super) _entry: Entry,
    pub(super) instance: ash::Instance,
    pub(super) device: ash::Device,
    pub(super) queue: vk::Queue,
    pub(super) command_pool: vk::CommandPool,
    pub(super) memory_properties: vk::PhysicalDeviceMemoryProperties,
    pub(super) device_name: String,
}

impl Drop for VkInner {
    fn drop(&mut self) {
        unsafe {
            let _ = self.device.device_wait_idle();
            self.device.destroy_command_pool(self.command_pool, None);
            self.device.destroy_device(None);
            self.instance.destroy_instance(None);
        }
    }
}

pub(super) struct Buffer {
    inner: Arc<VkInner>,
    pub(super) handle: vk::Buffer,
    memory: vk::DeviceMemory,
    pub(super) size: u64,
}

impl Drop for Buffer {
    fn drop(&mut self) {
        unsafe {
            self.inner.device.destroy_buffer(self.handle, None);
            self.inner.device.free_memory(self.memory, None);
        }
    }
}

pub(super) struct VkContext {
    pub(super) inner: Arc<VkInner>,
    pub(super) fence: vk::Fence,
    pub(super) cmd: vk::CommandBuffer,
    pub(super) dummy: Buffer,
}

impl Drop for VkContext {
    fn drop(&mut self) {
        unsafe {
            let _ = self.inner.device.device_wait_idle();
            self.inner.device.destroy_fence(self.fence, None);
        }
    }
}

impl VkContext {
    pub(super) fn new(device_index: usize) -> Result<Self> {
        let entry = unsafe { Entry::load() }
            .map_err(|e| failure(format!("failed to load Vulkan library: {e}")))?;
        let app_info = vk::ApplicationInfo::default()
            .application_name(c"palito")
            .api_version(vk::make_api_version(0, 1, 1, 0));
        let create_info = vk::InstanceCreateInfo::default().application_info(&app_info);
        let instance = unsafe { entry.create_instance(&create_info, None) }
            .map_err(|e| failure(format!("vkCreateInstance failed: {e}")))?;

        let physical_devices = unsafe { instance.enumerate_physical_devices() }
            .map_err(|e| failure(format!("vkEnumeratePhysicalDevices failed: {e}")))?;
        if physical_devices.is_empty() {
            unsafe { instance.destroy_instance(None) };
            return Err(failure("no Vulkan GPU devices found"));
        }

        // Prefer discrete GPUs first, then integrated, then others, preserving order within each class.
        let mut ranked: Vec<(u8, usize, vk::PhysicalDevice, String)> = physical_devices
            .into_iter()
            .enumerate()
            .filter_map(|(idx, pd)| {
                let props = unsafe { instance.get_physical_device_properties(pd) };
                let queues = unsafe { instance.get_physical_device_queue_family_properties(pd) };
                let has_compute = queues
                    .iter()
                    .any(|q| q.queue_flags.contains(vk::QueueFlags::COMPUTE));
                if !has_compute {
                    return None;
                }
                let rank = match props.device_type {
                    vk::PhysicalDeviceType::DISCRETE_GPU => 0,
                    vk::PhysicalDeviceType::INTEGRATED_GPU => 1,
                    vk::PhysicalDeviceType::VIRTUAL_GPU => 2,
                    _ => 3,
                };
                let name = unsafe { CStr::from_ptr(props.device_name.as_ptr()) }
                    .to_string_lossy()
                    .into_owned();
                Some((rank, idx, pd, name))
            })
            .collect();
        ranked.sort_by_key(|&(rank, idx, _, _)| (rank, idx));
        let (_, _, physical_device, device_name) = ranked
            .into_iter()
            .nth(device_index)
            .ok_or_else(|| failure(format!("Vulkan device index {device_index} out of range")))?;

        let queue_families =
            unsafe { instance.get_physical_device_queue_family_properties(physical_device) };
        let queue_family = queue_families
            .iter()
            .position(|q| q.queue_flags.contains(vk::QueueFlags::COMPUTE))
            .ok_or_else(|| failure("no compute queue family on Vulkan device"))?
            as u32;

        let priorities = [1.0f32];
        let queue_info = [vk::DeviceQueueCreateInfo::default()
            .queue_family_index(queue_family)
            .queue_priorities(&priorities)];
        let device_info = vk::DeviceCreateInfo::default().queue_create_infos(&queue_info);
        let device = unsafe { instance.create_device(physical_device, &device_info, None) }
            .map_err(|e| failure(format!("vkCreateDevice failed: {e}")))?;
        let queue = unsafe { device.get_device_queue(queue_family, 0) };
        let memory_properties =
            unsafe { instance.get_physical_device_memory_properties(physical_device) };

        let pool_info = vk::CommandPoolCreateInfo::default()
            .queue_family_index(queue_family)
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);
        let command_pool = unsafe { device.create_command_pool(&pool_info, None) }
            .map_err(|e| failure(format!("vkCreateCommandPool failed: {e}")))?;

        let cmd_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(command_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);
        let cmd = unsafe { device.allocate_command_buffers(&cmd_info) }
            .map_err(|e| failure(format!("vkAllocateCommandBuffers failed: {e}")))?[0];

        let fence_info = vk::FenceCreateInfo::default();
        let fence = unsafe { device.create_fence(&fence_info, None) }
            .map_err(|e| failure(format!("vkCreateFence failed: {e}")))?;

        let inner = Arc::new(VkInner {
            _entry: entry,
            instance,
            device,
            queue,
            command_pool,
            memory_properties,
            device_name,
        });
        let dummy = Self::alloc_buffer_inner(
            &inner,
            16,
            vk::BufferUsageFlags::STORAGE_BUFFER
                | vk::BufferUsageFlags::TRANSFER_DST
                | vk::BufferUsageFlags::TRANSFER_SRC,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        )?;
        Ok(Self {
            inner,
            fence,
            cmd,
            dummy,
        })
    }

    pub(super) fn device_name(&self) -> &str {
        &self.inner.device_name
    }

    fn find_memory_type(
        props: &vk::PhysicalDeviceMemoryProperties,
        type_bits: u32,
        required: vk::MemoryPropertyFlags,
    ) -> Option<u32> {
        for i in 0..props.memory_type_count {
            let ok_type = (type_bits & (1 << i)) != 0;
            let ok_flags = props.memory_types[i as usize]
                .property_flags
                .contains(required);
            if ok_type && ok_flags {
                return Some(i);
            }
        }
        None
    }

    fn alloc_buffer_inner(
        inner: &Arc<VkInner>,
        size: u64,
        usage: vk::BufferUsageFlags,
        preferred: vk::MemoryPropertyFlags,
    ) -> Result<Buffer> {
        let size = size.max(4).div_ceil(4) * 4;
        let info = vk::BufferCreateInfo::default()
            .size(size)
            .usage(usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);
        let handle = unsafe { inner.device.create_buffer(&info, None) }
            .map_err(|e| failure(format!("vkCreateBuffer({size}) failed: {e}")))?;
        let reqs = unsafe { inner.device.get_buffer_memory_requirements(handle) };
        let memory_type =
            Self::find_memory_type(&inner.memory_properties, reqs.memory_type_bits, preferred)
                .or_else(|| {
                    Self::find_memory_type(
                        &inner.memory_properties,
                        reqs.memory_type_bits,
                        vk::MemoryPropertyFlags::HOST_VISIBLE
                            | vk::MemoryPropertyFlags::HOST_COHERENT,
                    )
                })
                .ok_or_else(|| failure("no suitable Vulkan memory type"))?;
        let alloc_info = vk::MemoryAllocateInfo::default()
            .allocation_size(reqs.size)
            .memory_type_index(memory_type);
        let memory = unsafe { inner.device.allocate_memory(&alloc_info, None) }
            .map_err(|e| {
                unsafe { inner.device.destroy_buffer(handle, None) };
                failure(format!("vkAllocateMemory({}) failed: {e}", reqs.size))
            })?;
        unsafe { inner.device.bind_buffer_memory(handle, memory, 0) }.map_err(|e| {
            unsafe {
                inner.device.destroy_buffer(handle, None);
                inner.device.free_memory(memory, None);
            }
            failure(format!("vkBindBufferMemory failed: {e}"))
        })?;
        Ok(Buffer {
            inner: inner.clone(),
            handle,
            memory,
            size,
        })
    }

    pub(super) fn alloc_zeros(&self, size: u64) -> Result<Buffer> {
        let buf = Self::alloc_buffer_inner(
            &self.inner,
            size,
            vk::BufferUsageFlags::STORAGE_BUFFER
                | vk::BufferUsageFlags::TRANSFER_DST
                | vk::BufferUsageFlags::TRANSFER_SRC,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        )?;
        self.submit_one_shot(|device, cmd| unsafe {
            device.cmd_fill_buffer(cmd, buf.handle, 0, buf.size, 0);
        })?;
        Ok(buf)
    }

    pub(super) fn upload_bytes(&self, data: &[u8]) -> Result<Buffer> {
        let size = (data.len() as u64).max(4).div_ceil(4) * 4;
        let staging = Self::alloc_buffer_inner(
            &self.inner,
            size,
            vk::BufferUsageFlags::TRANSFER_SRC,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )?;
        unsafe {
            let ptr = self
                .inner
                .device
                .map_memory(staging.memory, 0, size, vk::MemoryMapFlags::empty())
                .map_err(|e| failure(format!("vkMapMemory failed: {e}")))?
                as *mut u8;
            std::ptr::copy_nonoverlapping(data.as_ptr(), ptr, data.len());
            if (data.len() as u64) < size {
                std::ptr::write_bytes(ptr.add(data.len()), 0, (size as usize) - data.len());
            }
            self.inner.device.unmap_memory(staging.memory);
        }
        let dst = Self::alloc_buffer_inner(
            &self.inner,
            size,
            vk::BufferUsageFlags::STORAGE_BUFFER
                | vk::BufferUsageFlags::TRANSFER_DST
                | vk::BufferUsageFlags::TRANSFER_SRC,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        )?;
        self.submit_one_shot(|device, cmd| unsafe {
            let region = [vk::BufferCopy::default().size(size)];
            device.cmd_copy_buffer(cmd, staging.handle, dst.handle, &region);
        })?;
        Ok(dst)
    }

    pub(super) fn alloc_host_visible(&self, size: u64) -> Result<Buffer> {
        Self::alloc_buffer_inner(
            &self.inner,
            size,
            vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::TRANSFER_SRC,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )
    }

    pub(super) fn read_mapped_f32(&self, staging: &Buffer, dst: &mut [f32]) -> Result<()> {
        let bytes = (dst.len() * 4) as u64;
        if bytes == 0 {
            return Ok(());
        }
        unsafe {
            let ptr = self
                .inner
                .device
                .map_memory(staging.memory, 0, bytes, vk::MemoryMapFlags::empty())
                .map_err(|e| failure(format!("vkMapMemory failed: {e}")))?
                as *const f32;
            std::ptr::copy_nonoverlapping(ptr, dst.as_mut_ptr(), dst.len());
            self.inner.device.unmap_memory(staging.memory);
        }
        Ok(())
    }

    pub(super) fn submit_one_shot(
        &self,
        record: impl FnOnce(&ash::Device, vk::CommandBuffer),
    ) -> Result<()> {
        let device = &self.inner.device;
        unsafe {
            device
                .reset_command_buffer(self.cmd, vk::CommandBufferResetFlags::empty())
                .map_err(|e| failure(format!("vkResetCommandBuffer failed: {e}")))?;
            let begin = vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
            device
                .begin_command_buffer(self.cmd, &begin)
                .map_err(|e| failure(format!("vkBeginCommandBuffer failed: {e}")))?;
            record(device, self.cmd);
            device
                .end_command_buffer(self.cmd)
                .map_err(|e| failure(format!("vkEndCommandBuffer failed: {e}")))?;
            device
                .reset_fences(&[self.fence])
                .map_err(|e| failure(format!("vkResetFences failed: {e}")))?;
            let cmds = [self.cmd];
            let submit = [vk::SubmitInfo::default().command_buffers(&cmds)];
            device
                .queue_submit(self.inner.queue, &submit, self.fence)
                .map_err(|e| failure(format!("vkQueueSubmit failed: {e}")))?;
            device
                .wait_for_fences(&[self.fence], true, u64::MAX)
                .map_err(|e| failure(format!("vkWaitForFences failed: {e}")))?;
        }
        Ok(())
    }
}
