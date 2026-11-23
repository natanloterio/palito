//! WGSL -> SPIR-V shader compilation via `naga` and Vulkan compute pipeline management.

use super::device::{Buffer, VkInner, failure};
use crate::Result;
use ash::vk;
use std::{collections::HashMap, ffi::CString, sync::Arc};

const SHADER_SOURCE: &str = include_str!("shaders.wgsl");

const ENTRY_POINTS: &[&str] = &[
    "embedding",
    "embedding_batch",
    "rmsnorm",
    "rmsnorm_batch",
    "matvec",
    "matmul_batch",
    "add_row_bias",
    "add_row_bias_batch",
    "head_norm",
    "head_norm_batch",
    "rotary",
    "rotary_batch",
    "cache_kv",
    "cache_kv_batch",
    "attention",
    "attention_batch",
    "swiglu",
    "residual",
    "scale_vec",
    "softcap",
    "router",
    "combine_experts",
    "copy_last_token",
];

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub(super) struct PushParams {
    pub p: [u32; 8],
    pub f: [f32; 4],
}

impl PushParams {
    pub(super) fn as_bytes(&self) -> &[u8] {
        unsafe {
            std::slice::from_raw_parts(
                (self as *const Self).cast::<u8>(),
                std::mem::size_of::<Self>(),
            )
        }
    }
}

pub(super) struct Pipelines {
    inner: Arc<VkInner>,
    module: vk::ShaderModule,
    set_layout: vk::DescriptorSetLayout,
    pub(super) layout: vk::PipelineLayout,
    descriptor_pool: vk::DescriptorPool,
    pipelines: HashMap<&'static str, vk::Pipeline>,
}

impl Drop for Pipelines {
    fn drop(&mut self) {
        unsafe {
            let device = &self.inner.device;
            let _ = device.device_wait_idle();
            for &pipeline in self.pipelines.values() {
                device.destroy_pipeline(pipeline, None);
            }
            device.destroy_descriptor_pool(self.descriptor_pool, None);
            device.destroy_pipeline_layout(self.layout, None);
            device.destroy_descriptor_set_layout(self.set_layout, None);
            device.destroy_shader_module(self.module, None);
        }
    }
}

impl Pipelines {
    pub(super) fn compile(inner: Arc<VkInner>, max_dispatches_per_token: u32) -> Result<Self> {
        let wgsl = naga::front::wgsl::parse_str(SHADER_SOURCE)
            .map_err(|e| failure(format!("WGSL parse error: {}", e.emit_to_string(SHADER_SOURCE))))?;
        let mut validator = naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::PUSH_CONSTANT,
        );
        let module_info = validator
            .validate(&wgsl)
            .map_err(|e| failure(format!("WGSL validation error: {e}")))?;
        let spv_options = naga::back::spv::Options {
            lang_version: (1, 3),
            flags: naga::back::spv::WriterFlags::ADJUST_COORDINATE_SPACE,
            ..Default::default()
        };
        let words = naga::back::spv::write_vec(&wgsl, &module_info, &spv_options, None)
            .map_err(|e| failure(format!("SPIR-V codegen error: {e}")))?;

        let device = &inner.device;
        let shader_info = vk::ShaderModuleCreateInfo::default().code(&words);
        let module = unsafe { device.create_shader_module(&shader_info, None) }
            .map_err(|e| failure(format!("vkCreateShaderModule failed: {e}")))?;

        let bindings: Vec<vk::DescriptorSetLayoutBinding<'_>> = (0..6u32)
            .map(|binding| {
                vk::DescriptorSetLayoutBinding::default()
                    .binding(binding)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .descriptor_count(1)
                    .stage_flags(vk::ShaderStageFlags::COMPUTE)
            })
            .collect();
        let set_layout_info = vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);
        let set_layout = unsafe { device.create_descriptor_set_layout(&set_layout_info, None) }
            .map_err(|e| failure(format!("vkCreateDescriptorSetLayout failed: {e}")))?;

        let push_range = [vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::COMPUTE)
            .offset(0)
            .size(std::mem::size_of::<PushParams>() as u32)];
        let set_layouts = [set_layout];
        let layout_info = vk::PipelineLayoutCreateInfo::default()
            .set_layouts(&set_layouts)
            .push_constant_ranges(&push_range);
        let layout = unsafe { device.create_pipeline_layout(&layout_info, None) }
            .map_err(|e| failure(format!("vkCreatePipelineLayout failed: {e}")))?;

        let max_sets = max_dispatches_per_token.max(256);
        let pool_sizes = [vk::DescriptorPoolSize::default()
            .ty(vk::DescriptorType::STORAGE_BUFFER)
            .descriptor_count(max_sets * 6)];
        let pool_info = vk::DescriptorPoolCreateInfo::default()
            .max_sets(max_sets)
            .pool_sizes(&pool_sizes);
        let descriptor_pool = unsafe { device.create_descriptor_pool(&pool_info, None) }
            .map_err(|e| failure(format!("vkCreateDescriptorPool failed: {e}")))?;

        let mut pipelines = HashMap::with_capacity(ENTRY_POINTS.len());
        for &name in ENTRY_POINTS {
            let cname = CString::new(name).unwrap();
            let stage = vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::COMPUTE)
                .module(module)
                .name(&cname);
            let info = [vk::ComputePipelineCreateInfo::default()
                .stage(stage)
                .layout(layout)];
            let pipeline = unsafe {
                device.create_compute_pipelines(vk::PipelineCache::null(), &info, None)
            }
            .map_err(|(_, e)| failure(format!("vkCreateComputePipelines({name}) failed: {e}")))?[0];
            pipelines.insert(name, pipeline);
        }

        Ok(Self {
            inner,
            module,
            set_layout,
            layout,
            descriptor_pool,
            pipelines,
        })
    }

    pub(super) fn reset_descriptors(&self) -> Result<()> {
        unsafe {
            self.inner
                .device
                .reset_descriptor_pool(self.descriptor_pool, vk::DescriptorPoolResetFlags::empty())
                .map_err(|e| failure(format!("vkResetDescriptorPool failed: {e}")))
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn dispatch(
        &self,
        cmd: vk::CommandBuffer,
        name: &str,
        buffers: [&Buffer; 6],
        params: PushParams,
        grid: (u32, u32, u32),
    ) -> Result<()> {
        let device = &self.inner.device;
        let pipeline = *self
            .pipelines
            .get(name)
            .ok_or_else(|| failure(format!("unknown Vulkan pipeline {name}")))?;
        let layouts = [self.set_layout];
        let alloc_info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(self.descriptor_pool)
            .set_layouts(&layouts);
        let set = unsafe { device.allocate_descriptor_sets(&alloc_info) }
            .map_err(|e| failure(format!("vkAllocateDescriptorSets failed: {e}")))?[0];

        let infos: [vk::DescriptorBufferInfo; 6] = std::array::from_fn(|i| {
            vk::DescriptorBufferInfo::default()
                .buffer(buffers[i].handle)
                .offset(0)
                .range(buffers[i].size)
        });
        let writes: [vk::WriteDescriptorSet<'_>; 6] = std::array::from_fn(|i| {
            vk::WriteDescriptorSet::default()
                .dst_set(set)
                .dst_binding(i as u32)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(std::slice::from_ref(&infos[i]))
        });
        unsafe {
            device.update_descriptor_sets(&writes, &[]);
            device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, pipeline);
            device.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::COMPUTE,
                self.layout,
                0,
                &[set],
                &[],
            );
            device.cmd_push_constants(
                cmd,
                self.layout,
                vk::ShaderStageFlags::COMPUTE,
                0,
                params.as_bytes(),
            );
            device.cmd_dispatch(cmd, grid.0, grid.1, grid.2);
            let barrier = [vk::MemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                .dst_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE)];
            device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::DependencyFlags::empty(),
                &barrier,
                &[],
                &[],
            );
        }
        Ok(())
    }
}
