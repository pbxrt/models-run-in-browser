<template>
  <input v-model="text" placeholder="Enter text here" />
</template>
<script setup lang="ts">
import { ref } from 'vue';
import { AutoModel, AutoProcessor, env } from '@xenova/transformers';
const text = ref('');

onMounted(() => {

})

</script>
