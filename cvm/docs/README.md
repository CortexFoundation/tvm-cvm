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

#### Currently Supported Operators

| Operator      | Supported          | Operator          | Supported          |
| ------------- | ------------------ | ----------------- | ------------------ |
| SliceAxis     | :heavy_check_mark: | Convolution       | :heavy_check_mark: |
| Slice         | :heavy_check_mark: | Pad               | :heavy_check_mark: |
| SliceLike     | :x:                | Expand_dims       | :heavy_check_mark: |
| Transpose     | :heavy_check_mark: | Embedding         | :heavy_check_mark: |
| relu          | :heavy_check_mark: | repeat            | :heavy_check_mark: |
| LeakyReLU     | :x:                | _contrib_box_nms  | :x:                |
| _mul_scalar   | :x:                | SliceChannel      | :x:                |
| _div_scalar   | :x:                | UpSampling        | :x:                |
| Activation    | :heavy_check_mark: | FullyConnected    | :heavy_check_mark: |
| sigmoid       | :heavy_check_mark: | broadcast_div     | :x:                |
| exp           | :heavy_check_mark: | broadcast_sub     | :x:                |
| softmax       | :heavy_check_mark: | broadcast_to      | :x:                |
| Pooling       | :heavy_check_mark: | broadcast_greater | :x:                |
| broadcast_mul | :heavy_check_mark: | Concat            | :heavy_check_mark: |
| broadcast_add | :heavy_check_mark: | sum               | :heavy_check_mark: |
| BatchNorm     | :x:                | ceil              | :x:                |
| Flatten       | :heavy_check_mark: | round             | :x:                |
| floor         | :x:                | fix               | :x:                |
| Cast          | :x:                | clip              | :heavy_check_mark: |
| Reshape       | :heavy_check_mark: | _minimum          | :x:                |
| Custom        | :x:                | _maximum          | :heavy_check_mark: |
| max           | :heavy_check_mark: | min               | :x:                |
| argmax        | :x:                | argmin            | :x:                |
| abs           | :x:                | elemwise_add      | :heavy_check_mark: |
| elemwise_sub  | :heavy_check_mark: | Dropout           | :heavy_check_mark: |
| _arange       | :heavy_check_mark: | tile              | :heavy_check_mark: |
| negative      | :heavy_check_mark: | SwapAxis          | :x:                |
| _plus_scalar  | :x:                | zeros_like        | :x:                |
| ones_like     | :x:                | _greater_scalar   | :x:                |
| where         | :x:                | squeeze           | :heavy_check_mark: |


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

#### MRT

A wrapper class for model transformation tool which simulates deep learning network integer computation within a float-point context. Model calibration and quantization are performed based on a specified model. This class has wrapped some user-friendly functions API introduced as below.

| func name                        | usage                                                        |
| -------------------------------- | ------------------------------------------------------------ |
| set_data(data)                   | Set the data before calibration.                             |
| calibrate([ctx, lambd, old_ths]) | Calibrate the current model after setting mrt data.<br />Contex on which intermediate result would be stored, hyperparameter lambd and reference threshold dict could also be specified. <br />Return the threshold dict of node-level output. |
| set_threshold(name, threshold)   | Manually set the threshold of the node output, given node name. |
| set_th_dict(th_dict)             | Manually set the threshold dict.                             |
| set_input_prec(prec)             | Set the input precision before quantization.                 |
| set_out_prec(prec)               | Set the output precision before quantization.                |
| set_softmax_lambd(val)           | Set the hyperparameter softmax_lambd before quantization.    |
| set_shift_bits(val)              | Set the hyperparameter shift_bits before quantization.       |
| quantize()                       | Quantize the current model after calibration.<br />Return the quantized model. |
| get_output_scales()              | Get the output scale of the model after quantization.        |
| get_maps()                       | Get the current name to old name map of the outputs after calibration or quantization. |
| get_inputs_ext()                 | Get the input_ext of the input after quantization.           |
| save(model_name[, datadir])      | save the current mrt instance into disk.                     |
| load(model_name[, datadir])      | [**staticmethod**]Return the mrt instance.<br />The given path should contain corresponding '.json' and '.params' file storing model information and '.ext' file storing mrt information. |



#### ModelMerger

A wrapper class for model merge tool. This class has wrapped some user-friendly functions API introduced as below.

| func name                             | usage                                                        |
| ------------------------------------- | ------------------------------------------------------------ |
| merge([callback])                     | Return the merged model. <br />Callback function could also be specified for updating the top node attributes. |
| get_output_scales(base_oscales, maps) | Get the model output scales after merge.<br />Base model output scales and base name maps should be specified. |







