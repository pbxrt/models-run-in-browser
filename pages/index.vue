<template>
  <div>
    <h1>Welcome to the homepage</h1>
    <ClientOnly>
      <OnnxModel />
    </ClientOnly>
  </div>
</template>
<script setup lang="ts">
import { ref, onMounted } from 'vue';
import OnnxModel from '@/components/OnnxModel.vue';

</script>
