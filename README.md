# Road Detection From Panchromatic Satellite Images Using Various Deep-Learning Models
<div style="text-align: justify">
In numerous civilian and military applications, the extraction of map items, such as roads,
rivers, and buildings, from high-resolution satellite images is a crucial undertaking. In this
project, pixel-wise binary segmentation in remote sensing images is used to compare various
deep-learning models for road detection. The dataset creation involves pairing original
panchromatic images with manually labelled binary masks depicting the roads along with
the augmentation of the entire dataset to increase sample diversity. The selected models
include our custom model RoadSegNN [with ResNet-50, ResNet-101, and Swin-T
backbones, along with an FPN, and custom Prediction module (Head)] and SegNet, which
are trained to perform binary segmentation of roads. Run-time evaluation and qualitative
and quantitative indicators are used in validation to evaluate computing efficiency and
prediction accuracy. In order to solve the binary road segmentation problem, the study offers
insights into the performance-specific subtleties of each model (and their variants) taken
into consideration.<br/><br/>

For this project four models have been selected:
- RoadSegNN (with ResNet-50 backbone)
- RoadSegNN (with ResNet-101 backbone)
- RoadSegNN (with Swin-T backbone)
- SegNet (Modified for our use case)
</div>


## Architecture of RoadSegNN Model (Novel Model)
<!-- <img src="./README_assets/RoadSegNN%20Architecture.jpg" width="74%" height = "74%">   -->
Printout is being carried.
<hr/>

## Architecture of ResNet-50 & ResNet-101
<p float="left">
  <img src="./README_assets/ResNet-50%20Architecture.jpg" width="40%" height = "50%">
  <img src="./README_assets/ResNet-101%20Architecture.jpg"  width="40.64%" height = "50%"> 
</p>    
<hr/>

## Architecture of SegNet Model
<img src="./README_assets/SegNet%20Architecture.png"  height = "68%">
<hr/>

## Sample Outputs
### Example 1:
<center>
    <table style="margin-bottom:0;">
        <tr>
            <th colspan=2><center>Input Image</center></th>
            <th><center>Output - SegNet</center></th>
        </tr>
        <tr>
            <td colspan=2><center><img src="./README_assets/1_input.jpg" width="245" height="245"></center></td>
            <td><center><img src="./README_assets/1_SegNet.png" width="245" height="245"></center></td>
        </tr>
        <tr>
            <th><center>Output - RoadSegNN (ResNet-50 backbone)</center></th>
            <th><center>Output - RoadSegNN (ResNet-101 backbone)</center></th>
            <th><center>Output - RoadSegNN (Swin-T backbone)</center></th>
        </tr>
        <tr>
            <td><center><img src="./README_assets/1_ResNet-50.png" width="245" height="245"></center></td>
            <td><center><img src="./README_assets/1_ResNet-101.png" width="245" height="245"></center></td>
            <td><center><img src="./README_assets/1_Swin-T.png" width="245" height="245" ></center></td>
        </tr>
    </table>
</center>
<hr/>

### Example 2:
<center>
    <table style="margin-bottom:0;">
        <tr>
            <th colspan=2><center>Input Image</center></th>
            <th><center>Output - SegNet</center></th>
        </tr>
        <tr>
            <td colspan=2><center><img src="./README_assets/32_input.jpg" width="245" height="245"></center></td>
            <td><center><img src="./README_assets/32_SegNet.png" width="245" height="245"></center></td>
        </tr>
        <tr>
            <th><center>Output - RoadSegNN (ResNet-50 backbone)</center></th>
            <th><center>Output - RoadSegNN (ResNet-101 backbone)</center></th>
            <th><center>Output - RoadSegNN (Swin-T backbone)</center></th>
        </tr>
        <tr>
            <td><center><img src="./README_assets/32_ResNet-50.png" width="245" height="245"></center></td>
            <td><center><img src="./README_assets/32_ResNet-101.png" width="245" height="245"></center></td>
            <td><center><img src="./README_assets/32_Swin-T.png" width="245" height="245" ></center></td>
        </tr>
    </table>
</center>