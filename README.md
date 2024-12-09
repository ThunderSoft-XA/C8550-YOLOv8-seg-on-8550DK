# C8550：YOLOv8-seg on 8550DK

#### Env Setup

1. Download SNPE-2.13.0.230730 from https://softwarecenter.qualcomm.com/#/catalog/item/qualcomm_neural_processing_sdk

2. Install snpe

   ```shell
   cp -r ${PATH_TO_SNPE}/include/SNPE /usr/include
   cp ${PATH_TO_SNPE}/lib/aarch64-oe-linux-gcc11.2/* /usr/lib/
   cp ${PATH_TO_SNPE}/lib/hexagon-v73/unsigned/* /usr/lib/rfsa/adsp/
   ```

3. Install opencv

   ``` shell
   apt-get update
   apt-get install libopencv-dev
   ```

4. Build

   ``` shell
   apt-get install build-essential cmake
   cd 8550-YOLOv8-seg/
   mkdir build
   cd build
   cmake ../
   make -j7
   ```

5. Run

   ``` shell
   ./test
   ```

   