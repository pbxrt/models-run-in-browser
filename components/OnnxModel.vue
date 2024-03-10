<template>
  <div>
    <h2>{{ status }}</h2>
    <input type="file" @change="handleFile" />
    <div class="container" :class="[status]" ref="containerRef">
      <canvas ref="canvasRef"></canvas>
    </div>
  </div>
</template>
<script setup lang="ts">
import { ref, onMounted } from 'vue';
import { AutoModel, AutoProcessor, env, RawImage, type Processor, type PreTrainedModel } from '@xenova/transformers';

env.allowRemoteModels = true;

const containerRef = ref();
const canvasRef = ref();
let model: PreTrainedModel;
let processor: Processor;
const status = ref('Loading...')

onMounted(async () => {
  model = await AutoModel.from_pretrained('briaai/RMBG-1.4', {
    config: {
      model_type: 'custom'
    }
  });
  processor = await AutoProcessor.from_pretrained('briaai/RMBG-1.4', {
    config: {
      do_normalize: true,
      do_pad: false,
      do_rescale: true,
      do_resize: true,
      image_mean: [.5, .5, .5],
      feature_extractor_type: 'ImageFeatureExtractor',
      image_std: [1, 1, 1],
      resample: 2,
      rescale_factor: .00392156862745098,
      size: {
        width: 1024,
        height: 1024
      }
    }
  });
  status.value = 'Ready';
})

const handleFile = (event) => {
  const [file] = event.target.files;

  const fileReader = new FileReader;
  fileReader.onload = (result) => {
    predict(result.target?.result);
  };
  fileReader.readAsDataURL(file);
}

const predict = async (url: string) => {
  status.value = 'predicting...'
  const rawImg = await RawImage.fromURL(url);
  const ratio = rawImg.width / rawImg.height;
  const [w, h] = ratio > 720 / 480 ? [720, 720 / ratio] : [480 * ratio, 480];

  const { pixel_values } = await processor(rawImg);
  const { output } = await model({
    input: pixel_values
  });

  const rawImg2 = await RawImage.fromTensor(output[0].mul(255).to('uint8')).resize(rawImg.width, rawImg.height);
  const canvas = canvasRef.value;
  canvas.width = rawImg.width;
  canvas.height = rawImg.height;
  const ctx = canvas.getContext('2d');
  if (!ctx) {
    return;
  }
  ctx.drawImage(rawImg.toCanvas(), 0, 0);
  const rawImgData = ctx.getImageData(0, 0, rawImg.width, rawImg.height);

  for (let i = 0; i < rawImg2.data.length; i++) {
    rawImgData.data[4 * i + 3] = rawImg2.data[i]
  }

  ctx.putImageData(rawImgData, 0, 0);

  containerRef.value.appendChild(canvas);
  status.value = 'Done'
}

</script>
<style lang="scss" scoped>
.container {
  canvas {
    display: none;
    width: 400px;
    margin: 0 auto;
    background: url("data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABAAAAAQBAMAAADt3eJSAAAAAXNSR0IArs4c6QAAAARnQU1BAACxjwv8YQUAAAAGUExURb+/v////5nD/3QAAAAJcEhZcwAADsMAAA7DAcdvqGQAAAAUSURBVBjTYwABQSCglEENMxgYGAAynwRB8BEAgQAAAABJRU5ErkJggg==");
  }
  &.Done {
    canvas {
      display: block;
    }
  }
}
</style>

