## Model Performance

Some models have been compiled and the Accuracy and size of params is provided in the following chart for reference.

| Model Name                | Category       | (Top1) Accuracy<br />(original / quantized) |
| ------------------------- | -------------- | ------------------------------------------- |
| ssd_512_mobilenet1.0_coco | detection      | 21.50% / 15.60%                             |
| ssd_512_resnet50_v1_voc   | detection      | 80.27% / 80.01%                             |
| yolo3_darknet53_voc       | detection      | 81.37% / 82.08%                             |
| shufflenet_v1             | classification | 63.48% / 60.45%                             |
| mobilenet1_0              | classification | 70.77% / 66.11%                             |
| mobilenetv2_1.0           | classification | 71.51% / 69.39%                             |

| Model Name                | Params Size | Path                                    |
| ------------------------- | ----------- | --------------------------------------- |
| ssd_512_mobilenet1.0_coco | 23.2M       | /data/mrt/ssd_512_mobilenet1.0_coco_tfm |
| ssd_512_resnet50_v1_voc   | 36.4M       | /data/mrt/ssd_512_resnet50_v1_voc_tfm   |
| yolo3_darknet53_voc       | 59.3M       | /data/mrt/yolo3_darknet53_voc_tfm       |
| shufflenet_v1             | 1.8M        | /data/mrt/shufflenet_v1_tfm             |
| mobilenet1_0              | 4.1M        | /data/mrt/mobilenet1_0_tfm              |
| mobilenetv2_1.0           | 3.4M        | /data/mrt/mobilenetv2_1.0_tfm           |