# NIAAnnotationTools

## Introduction
한국 정보화 진흥원 2020 인공지능 데이터셋 어노테이션 툴 입니다

** 본 툴에서 제공하는 기능
 
 - video 파일 업로드 및 프레임 단위 이미지 분할
 - 2D Human pose estimation을 위한 2D Keypoint dataset 구축
 - 3D Human pose estimation을 위한 3D Keyopint dataset 구축
 - Human shape recovery를 위한 rotation, shape params(SMPL), trans params(SMPL)
 
 ## 구축 데이터셋 종류 및 형태
 
 1. 2D_json (file-naming: {franme_no}.json)
 ```
 info{
  "supercategory" : str,
  "img_width" : int,
  "img_height : int,
  "camera_no" : int,
  "2d_pos  : [2D-keypoint location info],
  "annotations" : [annotations]
 }
 ```
```
annotations {
 "img_no" : int,
 "img_path" : str,
 "2d_pos" : [2D-keypoint location]
}
```

2. 3D_json (file-naming: 3D_{franme_no}.json)
```
 info{
  "supercategory" : str,
  "img_width" : int,
  "img_height : int,
  "camera_no" : int,
  "3d_pos : [3D-keypoint location info],
  "3d_rot" : [3D-keyopint rotation info],
  "annotations" : [annotations]
 }
```
```
annotations{
 "frame_no" : str,
 "obj_path" : str,
 "3d_pos" : [3D-keypoint location],
 "3d_rot" : [3D-keypoint rotation],
 "trans_params" : [trans parameter(SMPL)]
}
```
3. Camera_json (file-naming: {camera_no}.json)
```
{
"camera_no" : int,
"extrinsics" : [extrinsics matrix],
"intrinsics" : [intrinsics matrix]
}
```
4. Shape_json (file-naming: {frame_no}.json)
```
{
 "shape_params" : [shape parameter(SMPL)]
}
```
