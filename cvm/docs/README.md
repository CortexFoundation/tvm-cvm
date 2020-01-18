# Model Representation Tool Documentation

## Configuration File API | User Interface

MRT has separated model quantization configurations from source code for simplifying the user-usage. So one can quantize themself model quickly via configuring the .ini file. The running command script is as below.

``` bash
python cvm/quantization/main2.py config/file/path
```

Please refer to the example file: cvm/quantization/docs/example.ini for more configuration details. Copy the example file and configure yourself model's quantization settings.

The unify quantization procedure is defined in file: cvm/quantization/main2.py, refer to [main2](https://github.com) for more quantization details.

## Developer API

Mainly public quantization API is in file cvm/quantization/transformer.py, see the detail interface in the following sections. And the main quantization procedure is: 

​	Model Load >>> Preparation >>> [Optional] Model Split >>>

​	Calibration >>> Quantization >>> [Optional] Model Merge >>> Compilation to CVM,

which maps the class methods: 

​	Model.load >>> Model.prepare >>> [Optional] Model.split >>> 

​	MRT.calibrate >>> MRT.quantize >>> [Optional] ModelMerger.merge >>> Model.to_cvm.

The Calibration and Quantization pass is achieved in class MRT.

### Split && Merge

MRT has supported lots of mxnet operators while there still exists some unsupported. And all the unsupported operators are unquantifiable. So we just advise splitting the model into two sub-graph if there are some unsupported operators and only quantizing the half model (named base_model, indicating the input nodes to split operators generally). In other words, it's the user's responsibility to select the split keys of splitting the original model, while the half model is ignored to quantization pass if necessary. 

The list operators have already been considered by MRT developers, which operators over the list is not allowed in quantization. Contact the MRT developers in the github for more help.

#### Currently Supported Operators (TODO)

| Operator  | Supported          | Operator | Supported          |
| --------- | ------------------ | -------- | ------------------ |
| SliceAxis | :heavy_check_mark: |          | :heavy_check_mark: |
| Slice     | :x:                |          | :heavy_check_mark: |
| SliceLike | :heavy_check_mark: |          | :heavy_check_mark: |
|           | :heavy_check_mark: |          | :heavy_check_mark: |
|           | :heavy_check_mark: |          | :heavy_check_mark: |
|           | :heavy_check_mark: |          | :heavy_check_mark: |
|           | :heavy_check_mark: |          | :heavy_check_mark: |
|           | :heavy_check_mark: |          | :heavy_check_mark: |
|           | :heavy_check_mark: |          | :heavy_check_mark: |


### Public Interface

#### Model

A wrapper class for mxnet symbol and params which indicates model. All the quantization passes return the class instance for unify representation. Besides, the class has wrapped some user-friendly functions API introduced as below.

| func name                                          | usage                                                        |
| -------------------------------------------------- | ------------------------------------------------------------ |
| input_names()                                      | List the model's input names.                                |
| output_names()/names()                             | List the model's output names.                               |
| to_graph([dtype, ctx])                             | A convenient method to create model runtime.<br />Returns mxnet.gluon.nn.SymbolBlock. |
| save(symbol_file, params_file)                     | Dump model to disk.                                          |
| load(symbol_file, params_file)                     | **[staticmethod]** Load model from disk.                     |
| split(keys)                                        | Split the model by `keys` of model internal names.<br />Returns two sub-graph Model instances. |
| merger(base, top[, base_name_maps])                | [**staticmethod**] Returns the ModelMerger with two Model instance. |
| prepare([input_shape])                             | Model preparation passes, do operator checks, operator fusing, operator rewrite, ...etc. |
| to_cvm(model_name[, datadir, input_shape, target]) | Compile current mxnet quantization model into CVM accepted JSON&BINARY format. |

#### MRT(TODO)

#### ModelMerger(TODO)







