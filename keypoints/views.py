# import library
from django.shortcuts import render
from django.views.generic import TemplateView, View
from django.http import JsonResponse, HttpResponse, FileResponse
from keypoints.forms import UploadFileForm
from keypoints.models import ImageDataset
from django.db.models import Q, F
from django.db.models import Count, Value, CharField
from django.db.models.functions import Concat
from ast import literal_eval
import base64
import json
import tarfile
import trimesh
import shutil
from utils.display_utils import *
import torch
from keypoints.run_model import run_model, get_init_joint_2d, run_model_single


logger = logging.getLogger(__name__)
image_root_path = 'data'

# Create your views here.
class AnnotationView(TemplateView):
    template_name = 'annotation_3d_kp.html'
    origin_size = (1080, 1920)

    def get(self, request, *args, **kwargs):
        rows = ImageDataset.objects.values('camera_no').annotate(
            camera = F('camera_no'),
            frame_cnt = Count('camera_no'),
        ).values('camera', 'frame_cnt')
        frame_cnt = min([x['frame_cnt'] for x in rows])

        frame_list = ['{}'.format(x) for x in range(frame_cnt)]

        logger.info('return:FileUploadView[GET]')
        return render(request, self.template_name, {'frame_list': frame_list})

    def post(self, request, *args, **kwargs):
        usage = request.POST['usage']
        if usage == 'getdata':
            frame_no = int(request.POST['frame_no'])
            width = int(request.POST['width'])

            rows = ImageDataset.objects.filter(frame_no=frame_no).values('camera_no', 'img_path').annotate(
                camera = F('camera_no'),
                image_path = Concat(F('camera_no'), Value('/'), F('img_path'), output_field=CharField()),
                joint_2d = F('joint_2d'),
                intrinsic = F('intrinsics'),
                extrinsic = F('extrinsics')
            ).values('camera', 'image_path', 'joint_2d', 'intrinsic', 'extrinsic')
            image_path_list = [(x['camera'], x['image_path'], x['joint_2d'], x['intrinsic'], x['extrinsic']) for x in rows]

            data = []
            for camera_no, image_path, joint2d, intrinsics, extrinsics in image_path_list:
                if joint2d is None:
                    try:
                        prev_joint = ImageDataset.objects.get(frame_no=frame_no-1, camera_no=camera_no).joint_2d
                    except:
                        prev_joint = None
                    if prev_joint is not None:
                        # init by previous frame pose
                        joint2d = prev_joint
                    else:
                        # init by T-pose
                        joint2d = str(get_init_joint_2d(
                            torch.Tensor(literal_eval(intrinsics)),
                            torch.Tensor(literal_eval(extrinsics)),
                            'neutral'
                        ).tolist())

                image = cv2.imread(osp.join(image_root_path, image_path))
                ratio = width / self.origin_size[1]
                dim = (width, int(self.origin_size[0] * ratio))
                joint2d = (np.array(literal_eval(joint2d))*ratio).tolist()
                resized_image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
                _, buffer = cv2.imencode('.jpg', resized_image)
                image_encoded = base64.b64encode(buffer).decode('utf-8')
                data.append({
                    'camera_no': camera_no,
                    'imagebytearray': image_encoded,
                    'joint2d': joint2d,
                    'overraybytearray': None
                })

            result = {
                'status': 'sucess',
                'data': data
            }
            response = result
        elif usage == 'save':
            frame_no = request.POST['frame_no']
            width = int(request.POST['width'])
            ratio = self.origin_size[1] / width

            datas = literal_eval(request.POST['data'])['data']
            for data in datas:
                camera_no = data['camera_no']
                joint2d = (np.array(data['joint2d'])*ratio).astype(np.int64).tolist()

                image_dataset = ImageDataset.objects.get(
                    frame_no=frame_no, camera_no=int(camera_no),
                )
                image_dataset.joint_2d = joint2d
                image_dataset.save()
            result = {
                'status': 'sucess',
                'data': data,
            }
            response = result
        elif usage == 'apply':
            frame_no = int(request.POST['frame_no'])
            width = int(request.POST['width'])

            rows = ImageDataset.objects.filter(frame_no=frame_no).values().annotate(
                camera = F('camera_no'),
                intrinsic = F('intrinsics'),
                extrinsic = F('extrinsics'),
                joint = F('joint_2d'),
                img_width = F('width'),
                img_height = F('height')
            ).values('camera', 'intrinsic', 'extrinsic', 'joint', 'img_width', 'img_height')
            gt_dataset = []

            for row in rows:
                gt_dataset.append([literal_eval(row['joint']), literal_eval(row['intrinsic']), literal_eval(row['extrinsic'])])
            mesh, Jtr, result = run_model(gt_dataset, 'neutral', 30)
            ImageDataset.objects.filter(frame_no=frame_no).update(**result)

            data = []
            for row in rows:
                # image = cv2.imread(osp.join(image_root_path, row['image_path']))
                image = np.zeros((row['img_height'], row['img_width'], 4))
                vis_image = display_model2(
                    image, mesh.squeeze().detach(),
                    torch.Tensor(literal_eval(row['intrinsic'])),
                    torch.Tensor(literal_eval(row['extrinsic'])),
                    self.origin_size[1], self.origin_size[0])

                ratio = width / self.origin_size[1]
                dim = (width, int(self.origin_size[0] * ratio))
                resized_image = cv2.resize(vis_image, dim, interpolation=cv2.INTER_AREA)
                _, buffer = cv2.imencode('.jpg', resized_image)
                image_encoded = base64.b64encode(buffer).decode('utf-8')
                data.append({
                    'camera_no': int(row['camera']),
                    'imagebytearray': None,
                    'joint2d': (np.array(literal_eval(row['joint']))*ratio).tolist(),
                    'overraybytearray': image_encoded
                })
            result = {
                'status': 'sucess',
                'data': data
            }
            response = result
        elif usage == 'download':
            frame_no = request.POST['frame_no']
            tar_list = []
            if not osp.exists('tmp'):
                os.mkdir('tmp')
            logger.info('make 2d joint json')
            datasets = ImageDataset.objects.filter(frame_no=frame_no).values().annotate(
                camera = F('camera_no'),
                image_path = Concat(F('camera_no'), Value('/'), F('img_path'), output_field=CharField()),
                joint = F('joint_2d'),
                img_width = F('width'),
                img_height = F('height'),
                image_no = F('img_no'),
            ).values('camera_no', 'image_path', 'joint', 'img_width', 'img_height', 'image_no')

            temp_path = osp.join('tmp', 'Image')
            if not osp.exists(temp_path):
                os.mkdir(temp_path)

            for data in datasets:
                info = {
                    'supercategory': 'Human',
                    'img_width': data['img_width'],
                    'img_height': data['img_height'],
                    'camera_no': data['camera_no'],
                    '2d_pos': [
                        'Pelvis','L_Hip','R_Hip','Spine1','L_Knee',
                        'R_Knee','Spine2','L_Ankle','R_Ankle','Spine3',
                        'L_Foot','R_Foot','Neck','L_Collar','R_Collar',
                        'Head','L_Shoulder','R_Shoulder','L_Elbow','R_Elbow',
                        'L_Wrist','R_Wrist','L_Hand','R_Hand'
                    ]
                }
                annotations = {
                    'img_no': data['image_no'],
                    'img_path': data['image_path'],
                    '2d_pos': literal_eval(data['joint'])
                }
                json_out = {
                    'info': info,
                    'annotations': annotations
                }
                filename = '{}_{}.json'.format(data['camera_no'], frame_no)
                temp_path = osp.join('tmp', '2D_json')
                if not osp.exists(temp_path):
                    os.mkdir(temp_path)
                with open(osp.join(temp_path, filename), 'w') as json_file:
                    json.dump(json_out, json_file, indent=4)

                tar_list.append(osp.join(temp_path, filename))
                temp_path = osp.join('tmp', 'Image')
                if not osp.exists(temp_path):
                    os.mkdir(temp_path)
                if not osp.exists(osp.join(temp_path, str(data['camera_no']))):
                    os.mkdir(osp.join(temp_path, str(data['camera_no'])))
                shutil.copy(osp.join(image_root_path, data['image_path']), osp.join(temp_path, str(data['camera_no'])))
                tar_list.append(osp.join(temp_path, str(data['camera_no'])))

            logger.info('make 3d joint json')
            data = ImageDataset.objects.filter(frame_no=frame_no).values().annotate(
                camera = F('camera_no'),
                image_path = Concat(F('camera_no'), Value('/'), F('img_path'), output_field=CharField()),
                joint = F('joint_3d'),
                rot = F('rotation'),
                img_width = F('width'),
                img_height = F('height'),
                image_no = F('img_no'),
                trans = F('trans_params'),
                shapes = F('shape_params')
            ).values('camera_no', 'image_path', 'joint', 'rot', 'img_width', 'img_height', 'image_no', 'trans', 'shapes')[0]
            info = {
                'supercategory': 'Human',
                'img_width': data['img_width'],
                'img_height': data['img_height'],
                'camera_no': data['camera_no'],
                '3d_pos': [
                    'Pelvis','L_Hip','R_Hip','Spine1','L_Knee',
                    'R_Knee','Spine2','L_Ankle','R_Ankle','Spine3',
                    'L_Foot','R_Foot','Neck','L_Collar','R_Collar',
                    'Head','L_Shoulder','R_Shoulder','L_Elbow','R_Elbow',
                    'L_Wrist','R_Wrist','L_Hand','R_Hand'
                ],
                '3d_rot': [
                    'Pelvis','L_Hip','R_Hip','Spine1','L_Knee',
                    'R_Knee','Spine2','L_Ankle','R_Ankle','Spine3',
                    'L_Foot','R_Foot','Neck','L_Collar','R_Collar',
                    'Head','L_Shoulder','R_Shoulder','L_Elbow','R_Elbow',
                    'L_Wrist','R_Wrist','L_Hand','R_Hand'
                ]
            }
            annotations = {
                'frame_no': frame_no,
                'obj_path': osp.join('3D_shape', '{}.obj'.format(frame_no)),
                '3d_pos': literal_eval(data['joint']),
                '3d_rot': literal_eval(data['rot']),
                'trans_params': literal_eval(data['trans'])
            }
            json_out = {
                'info': info,
                'annotations': annotations
            }
            filename = '3D_{}.json'.format(frame_no)
            temp_path = osp.join('tmp', '3D_json')
            if not osp.exists(temp_path):
                os.mkdir(temp_path)
            with open(osp.join(temp_path, filename), 'w') as json_file:
                json.dump(json_out, json_file, indent=4)
            tar_list.append(osp.join(temp_path, filename))

            logger.info('make obj')
            pose = np.array(literal_eval(data['rot']))
            vertice, faces = run_model_single(
                'neutral',
                pose_params=torch.from_numpy(
                    np.apply_along_axis(eulerAnglesToRotationMatrix, 1, pose).flatten()
                ),
                shape_params=torch.Tensor(literal_eval(data['shapes'])),
                trans_params=torch.Tensor(literal_eval(data['trans']))
            )
            mesh = trimesh.Trimesh(vertice, faces)
            temp_path = osp.join('tmp', '3D_shape')
            if not osp.exists(temp_path):
                os.mkdir(temp_path)
            filename = '{}.obj'.format(frame_no)
            save_obj(mesh.vertices, mesh.faces, osp.join(temp_path, filename))
            tar_list.append(osp.join(temp_path, filename))
            # make shape json
            logger.info('make shape json')
            temp_path = osp.join('tmp', 'Shape_params')
            if not osp.exists(temp_path):
                os.mkdir(temp_path)
            json_out = {
                'shape_param': literal_eval(data['shapes'])
            }
            filename = '{}.json'.format(frame_no)
            with open(osp.join(temp_path, filename), 'w') as json_file:
                json.dump(json_out, json_file, indent=4)
            tar_list.append(osp.join(temp_path, filename))
            logger.info('make camera json')
            temp_path = osp.join('tmp', 'Camera_json')
            if not osp.exists(temp_path):
                os.mkdir(temp_path)
            datasets = ImageDataset.objects.filter(frame_no=frame_no).values('camera_no', 'intrinsics', 'extrinsics')
            for data in datasets:
                json_out = {
                    'camera_no': data['camera_no'],
                    'extrinsics': literal_eval(data['extrinsics']),
                    'intrinsics': literal_eval(data['intrinsics'])
                }
                filename = '{}.json'.format(data['camera_no'])
                with open(osp.join(temp_path, filename), 'w') as json_file:
                    json.dump(json_out, json_file, indent=4)
                tar_list.append(osp.join(temp_path, filename))
            tar = tarfile.open(osp.join('tmp', 'annotation.tar.gz'), 'w:gz')
            for name in tar_list:
                tar.add(name)
            tar.close()
            logger.info('zip with .tar file')
            response = {
                'status': 'success',
                'filename': 'annotation.tar.gz'
            }
            for name in os.listdir('tmp'):
                path = osp.join('tmp', name)
                if osp.isdir(path):
                    shutil.rmtree(path)
        logger.info('return:FileUploadView[POSE;{}]'.format(usage))
        return JsonResponse(response, safe=False, json_dumps_params={'ensure_ascii': True})

def download(request, filepath):
    server_file_path = osp.join('tmp', filepath)
    fsock = open(server_file_path, 'rb')
    response = HttpResponse(fsock, content_type='application/zip')
    response['Content-Disposition'] = "attachment; filename=%s" % 'download.zip'
    return response

class FileUploadView(View):
    template_name = 'file_upload.html'
    thumbnail_width = 128
    n_camera = 4

    def get(self, request, *args, **kwargs):
        form = UploadFileForm()
        rows = ImageDataset.objects.filter(frame_no=0).values('camera_no').annotate(
            camera = F('camera_no'),
            image_path = Concat(F('camera_no'), Value('/'), F('img_path'), output_field=CharField()),
            intrinsic = F('intrinsics'),
            extrinsic = F('extrinsics')
            ).values('camera', 'image_path', 'intrinsic', 'extrinsic')

        saved_info = []
        if len(rows) < 1:
            for camera in range(self.n_camera):
                saved_info.append({
                    'camera': camera,
                    'src': "",
                    'intrinsics': np.zeros(9).tolist(),
                    'extrinsics': np.zeros(12).tolist()
                })
        else:
            for row in rows:
                camera = row['camera']
                image = cv2.imread(osp.join(image_root_path, row['image_path']))
                width = self.thumbnail_width
                ratio = width / image.shape[1]
                dim = (width, int(image.shape[0] * ratio))
                logger.info('make thumbnail image')
                thumbnail = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
                _, buffer = cv2.imencode('.jpg', thumbnail)
                src = base64.b64encode(buffer).decode('utf-8')
                saved_info.append({
                    'camera': camera,
                    'src': 'data:image/png;base64,' + src,
                    'intrinsics': np.array(literal_eval(row['intrinsic'])).flatten().tolist(),
                    'extrinsics': np.array(literal_eval(row['extrinsic'])).flatten().tolist()
                })
        for i in range(self.n_camera):
            is_contain = False
            for info in saved_info:
                if i == info['camera']:
                    is_contain = True
                    break
            if not is_contain:
                saved_info.append({'camera':i, 'src':''})
        logger.info('return:FileUploadView[GET]')
        return render(request, self.template_name, {'form': form, 'saved_info': saved_info})

    def post(self, request, *args, **kwargs):
        form = UploadFileForm(request.POST, request.FILES)
        if form.is_valid():
            camera_no = str(form.data['camera_no'])
            if not osp.exists(osp.join(image_root_path, camera_no)):
                os.makedirs(osp.join(image_root_path, camera_no), exist_ok=True)
            intrinsics = np.expand_dims(np.array(literal_eval('[{}]'.format(form.data['intrinsics']))), 0).reshape((3, 3)).tolist()
            extrinsics = np.expand_dims(np.array(literal_eval('[{}]'.format(form.data['extrinsics']))), 0).reshape((3, 4)).tolist()
            decoded_string = form.files['file'].read()
            # save temparary video
            logger.info('save temparary video')
            temp_path = 'tmp'
            if not osp.exists(temp_path):
                os.mkdir(temp_path)
            # read video for split to image by frame
            saved_path = osp.join(temp_path, 'video.mp4')
            with open(saved_path, 'wb') as wfile:
                wfile.write(decoded_string)
            vidcap = cv2.VideoCapture(saved_path)
            frame_no = 0
            success, image = vidcap.read()
            width = self.thumbnail_width
            ratio = width / image.shape[1]
            dim = (width, int(image.shape[0] * ratio))
            logger.info('make thumbnail image')
            thumbnail = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
            frame_no += 1
            logger.info('start split video frame...')
            while success:
                file_name = 'frame_{}.jpg'.format(frame_no)
                img_path = osp.join(image_root_path, camera_no, file_name)
                cv2.imwrite(img_path, image)
                success, image = vidcap.read()
                if not success:
                    break
                height, width, _ = image.shape
                image_dataset = ImageDataset(
                    img_path=file_name, frame_no=frame_no, camera_no=int(camera_no),
                    intrinsics=intrinsics, extrinsics=extrinsics, width=width, height=height
                )
                data = ImageDataset.objects.filter(Q(frame_no=frame_no), Q(camera_no=camera_no))
                if len(list(data)) > 0:
                    data.update(frame_no=frame_no, camera_no=camera_no,
                                intrinsics=intrinsics,
                                extrinsics=extrinsics,
                                width=width, height=height
                                )
                else:
                    image_dataset.save()
                frame_no += 1

            shutil.rmtree('tmp')
            logger.info('...end split video by frame')
            logger.info('encoding to base64')
            _, buffer = cv2.imencode('.jpg', thumbnail)
            data = base64.b64encode(buffer).decode('utf-8')
            result = {
                'status': 'sucess',
                'data': data
            }
            response = result
        else:
            response = {'status': 'false'}
        logger.info('return:FileUploadView[POSE]')
        return JsonResponse(response, safe=False, json_dumps_params={'ensure_ascii': True})
