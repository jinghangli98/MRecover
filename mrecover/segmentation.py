import torch
import nibabel
import numpy as np
import os
import scipy.ndimage
import torch.nn as nn
import torch.nn.functional as F
from numpy.linalg import inv

# Monkey-patch for older torch versions
try:
    import inspect
    if "align_corners" not in inspect.signature(F.grid_sample).parameters:
        old_grid_sample = F.grid_sample
        F.grid_sample = lambda *x, **k: old_grid_sample(*x)
except:
    pass

# Set device
device = torch.device("cpu")

# ============================================================================
# Model Definitions
# ============================================================================

class HeadModel(nn.Module):
    def __init__(self):
        super(HeadModel, self).__init__()
        self.conv0a = nn.Conv3d(1, 8, 3, padding=1)
        self.conv0b = nn.Conv3d(8, 8, 3, padding=1)
        self.bn0a = nn.BatchNorm3d(8)
        self.ma1 = nn.MaxPool3d(2)
        self.conv1a = nn.Conv3d(8, 16, 3, padding=1)
        self.conv1b = nn.Conv3d(16, 24, 3, padding=1)
        self.bn1a = nn.BatchNorm3d(24)
        self.ma2 = nn.MaxPool3d(2)
        self.conv2a = nn.Conv3d(24, 24, 3, padding=1)
        self.conv2b = nn.Conv3d(24, 32, 3, padding=1)
        self.bn2a = nn.BatchNorm3d(32)
        self.ma3 = nn.MaxPool3d(2)
        self.conv3a = nn.Conv3d(32, 48, 3, padding=1)
        self.conv3b = nn.Conv3d(48, 48, 3, padding=1)
        self.bn3a = nn.BatchNorm3d(48)
        self.conv2u = nn.Conv3d(48, 24, 3, padding=1)
        self.conv2v = nn.Conv3d(24+32, 24, 3, padding=1)
        self.bn2u = nn.BatchNorm3d(24)
        self.conv1u = nn.Conv3d(24, 24, 3, padding=1)
        self.conv1v = nn.Conv3d(24+24, 24, 3, padding=1)
        self.bn1u = nn.BatchNorm3d(24)
        self.conv0u = nn.Conv3d(24, 16, 3, padding=1)
        self.conv0v = nn.Conv3d(16+8, 8, 3, padding=1)
        self.bn0u = nn.BatchNorm3d(8)
        self.conv1x = nn.Conv3d(8, 4, 1, padding=0)

    def forward(self, x):
        x = F.elu(self.conv0a(x))
        self.li0 = x = F.elu(self.bn0a(self.conv0b(x)))
        x = self.ma1(x)
        x = F.elu(self.conv1a(x))
        self.li1 = x = F.elu(self.bn1a(self.conv1b(x)))
        x = self.ma2(x)
        x = F.elu(self.conv2a(x))
        self.li2 = x = F.elu(self.bn2a(self.conv2b(x)))
        x = self.ma3(x)
        x = F.elu(self.conv3a(x))
        x = F.elu(self.bn3a(self.conv3b(x)))
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        x = F.elu(self.conv2u(x))
        x = torch.cat([x, self.li2], 1)
        x = F.elu(self.bn2u(self.conv2v(x)))
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        x = F.elu(self.conv1u(x))
        x = torch.cat([x, self.li1], 1)
        x = F.elu(self.bn1u(self.conv1v(x)))
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        x = F.elu(self.conv0u(x))
        x = torch.cat([x, self.li0], 1)
        x = F.elu(self.bn0u(self.conv0v(x)))
        x = self.conv1x(x)
        return torch.sigmoid(x)


class ModelAff(nn.Module):
    def __init__(self):
        super(ModelAff, self).__init__()
        self.convaff1 = nn.Conv3d(2, 16, 3, padding=1)
        self.maaff1 = nn.MaxPool3d(2)
        self.convaff2 = nn.Conv3d(16, 16, 3, padding=1)
        self.bnaff2 = nn.LayerNorm([32, 32, 32])
        self.maaff2 = nn.MaxPool3d(2)
        self.convaff3 = nn.Conv3d(16, 32, 3, padding=1)
        self.bnaff3 = nn.LayerNorm([16, 16, 16])
        self.maaff3 = nn.MaxPool3d(2)
        self.convaff4 = nn.Conv3d(32, 64, 3, padding=1)
        self.maaff4 = nn.MaxPool3d(2)
        self.bnaff4 = nn.LayerNorm([8, 8, 8])
        self.convaff5 = nn.Conv3d(64, 128, 1, padding=0)
        self.convaff6 = nn.Conv3d(128, 12, 4, padding=0)

        gx, gy, gz = np.linspace(-1, 1, 64), np.linspace(-1, 1, 64), np.linspace(-1, 1, 64)
        grid = np.meshgrid(gx, gy, gz)
        grid = np.stack([grid[2], grid[1], grid[0], np.ones_like(grid[0])], axis=3)
        netgrid = np.swapaxes(grid, 0, 1)[..., [2, 1, 0, 3]]
        
        self.register_buffer('grid', torch.tensor(netgrid.astype("float32"), requires_grad=False))
        self.register_buffer('diagA', torch.eye(4, dtype=torch.float32))

    def forward(self, outc1):
        x = F.relu(self.convaff1(outc1))
        x = self.maaff1(x)
        x = F.relu(self.bnaff2(self.convaff2(x)))
        x = self.maaff2(x)
        x = F.relu(self.bnaff3(self.convaff3(x)))
        x = self.maaff3(x)
        x = F.relu(self.bnaff4(self.convaff4(x)))
        x = self.maaff4(x)
        x = F.relu(self.convaff5(x))
        x = self.convaff6(x)
        x = x.view(-1, 3, 4)
        x = torch.cat([x, x[:, 0:1] * 0], dim=1)
        self.tA = torch.transpose(x + self.diagA, 1, 2)
        wgrid = self.grid @ self.tA[:, None, None]
        gout = F.grid_sample(outc1, wgrid[..., [2, 1, 0]], align_corners=True)
        return gout, self.tA


class HippoModel(nn.Module):
    def __init__(self):
        super(HippoModel, self).__init__()
        self.conv0a_0 = nn.Conv3d(1, 16, (1, 1, 3), padding=0)
        self.conv0a_1 = nn.Conv3d(16, 16, (1, 3, 1), padding=0)
        self.conv0a = nn.Conv3d(16, 16, (3, 1, 1), padding=0)
        self.convf1 = nn.Conv3d(16, 48, (3, 3, 3), padding=0)
        self.maxpool1 = nn.MaxPool3d(2)
        self.bn1 = nn.BatchNorm3d(48, momentum=1)
        self.bn1.training = False
        self.convout0 = nn.Conv3d(48, 48, (3, 3, 3), padding=1)
        self.convout1 = nn.Conv3d(48, 48, (3, 3, 3), padding=1)
        self.maxpool2 = nn.MaxPool3d(2)
        self.bn2 = nn.BatchNorm3d(48, momentum=1)
        self.bn2.training = False
        self.convout2p = nn.Conv3d(48, 48, (3, 3, 3), padding=1)
        self.convout2 = nn.Conv3d(48, 48, (3, 3, 3), padding=1)
        self.convlx3 = nn.Conv3d(48, 48, (3, 3, 3), padding=1)
        self.convlx5 = nn.Conv3d(48, 48, (3, 3, 3), padding=1)
        self.convlx7 = nn.Conv3d(48, 16, (3, 3, 3), padding=1)
        self.convlx8 = nn.Conv3d(16, 1, 1, padding=0)
        self.blur = nn.Conv3d(1, 1, 7, padding=3)
        self.conv_extract = nn.Conv3d(48, 47, 3, padding=1)
        self.convmix = nn.Conv3d(48, 16, 3, padding=1)
        self.convout1x = nn.Conv3d(16, 1, 1, padding=0)

    def forward(self, x):
        x = F.relu(self.conv0a_0(x))
        x = F.relu(self.conv0a_1(x))
        x = F.relu(self.conv0a(x))
        self.out_conv_f1 = x = F.relu(self.convf1(x))
        self.out_maxpool1 = x = self.maxpool1(x)
        x = self.bn1(x)
        x = F.relu(self.convout0(x))
        x = self.convout1(x)
        x = x + self.out_maxpool1
        x = F.relu(x)
        self.out_maxpool2 = x = self.maxpool2(x)
        x = self.bn2(x)
        x = F.relu(self.convout2p(x))
        x = self.convout2(x)
        x = x + self.out_maxpool2
        x = F.relu(x)
        x = F.relu(self.convlx3(x))
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        x = F.relu(self.convlx5(x))
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        x = F.relu(self.convlx7(x))
        self.out_output1 = x = torch.sigmoid(self.convlx8(x))
        x = torch.sigmoid(self.blur(x))
        x = x * self.out_conv_f1
        x = F.leaky_relu(self.conv_extract(x))
        x = torch.cat([self.out_output1, x], dim=1)
        x = F.relu(self.convmix(x))
        x = torch.sigmoid(self.convout1x(x))
        return x


# ============================================================================
# Helper Functions
# ============================================================================

def bbox_world(affine, shape):
    s = shape[0]-1, shape[1]-1, shape[2]-1
    bbox = [[0,0,0], [s[0],0,0], [0,s[1],0], [0,0,s[2]], 
            [s[0],s[1],0], [s[0],0,s[2]], [0,s[1],s[2]], [s[0],s[1],s[2]]]
    w = affine @ np.column_stack([bbox, [1]*8]).T
    return w.T

bbox_one = np.array([[-1,-1,-1,1], [1,-1,-1,1], [-1,1,-1,1], [-1,-1,1,1], 
                     [1,1,-1,1], [1,-1,1,1], [-1,1,1,1], [1,1,1,1]])

affine64_mni = np.array([[-2.85714293, 0., 0., 90.],
                         [0., 3.42857146, 0., -126.],
                         [0., 0., 2.85714293, -72.],
                         [0., 0., 0., 1.]])

mul_homo = lambda g, Mt: g @ Mt[:3,:3].astype(np.float32) + Mt[3,:3].astype(np.float32)

def indices_unitary(dimensions, dtype):
    dimensions = tuple(dimensions)
    N = len(dimensions)
    shape = (1,)*N
    res = np.empty((N,)+dimensions, dtype=dtype)
    for i, dim in enumerate(dimensions):
        res[i] = np.linspace(-1, 1, dim, dtype=dtype).reshape(
            shape[:i] + (dim,) + shape[i+1:])
    return res

def bbox_xyz(shape, affine):
    s = shape[0]-1, shape[1]-1, shape[2]-1
    bbox = [[0,0,0], [s[0],0,0], [0,s[1],0], [0,0,s[2]], 
            [s[0],s[1],0], [s[0],0,s[2]], [0,s[1],s[2]], [s[0],s[1],s[2]]]
    return mul_homo(bbox, affine.T)

def indices_xyz(shape, affine, offset_vox=np.array([0,0,0])):
    ind = np.indices(shape).astype(np.float32) + offset_vox.reshape(3,1,1,1).astype(np.float32)
    return mul_homo(np.rollaxis(ind, 0, 4), affine.T)

def xyz_to_DHW3(xyz, iaffine, srcshape):
    affine = np.linalg.inv(iaffine)
    ijk3 = mul_homo(xyz, affine.T)
    ijk3[...,0] /= srcshape[0] - 1
    ijk3[...,1] /= srcshape[1] - 1
    ijk3[...,2] /= srcshape[2] - 1
    ijk3 = ijk3 * 2 - 1
    return np.swapaxes(ijk3, 0, 2)


# ============================================================================
# Load Models
# ============================================================================

scriptpath = os.path.dirname(os.path.realpath(__file__))

net = HeadModel().to(device)
net.load_state_dict(torch.load(
    scriptpath + "/torchparams/params_head_00075_00000.pt", 
    map_location=device))
net.eval()

netAff = ModelAff().to(device)
netAff.load_state_dict(torch.load(
    scriptpath + "/torchparams/paramsaffineta_00079_00000.pt", 
    map_location=device), strict=False)
netAff.eval()

hipponet = HippoModel().to(device)
hipponet.load_state_dict(torch.load(
    scriptpath + "/torchparams/hippodeep.pt", 
    map_location=device))
hipponet.eval()


# ============================================================================
# Main Segmentation Function
# ============================================================================

def segment_hippocampus(filename):
    """
    Segment left and right hippocampus from T1 MRI image.
    
    Args:
        filename: Path to T1 MRI image (nii.gz format)
        save_output: If True, saves output as NIfTI files (default: True)
    
    Returns:
        L_output: Left hippocampus segmentation (numpy array, uint8)
        R_output: Right hippocampus segmentation (numpy array, uint8)
    """
    # Load image
    img = nibabel.load(filename)
    if type(img) is nibabel.nifti1.Nifti1Image:
        img._affine = img.get_qform()
    
    d = img.get_fdata(caching="unchanged", dtype=np.float32)
    
    # Handle 4D images
    while len(d.shape) > 3:
        d = d.mean(-1)
    
    # Normalize
    d = (d - d.mean()) / d.std()
    
    # Reorient to LAS
    o1 = nibabel.orientations.io_orientation(img.affine)
    o2 = np.array([[0., -1.], [1., 1.], [2., 1.]])
    trn = nibabel.orientations.ornt_transform(o1, o2)
    trn_back = nibabel.orientations.ornt_transform(o2, o1)
    revaff1 = nibabel.orientations.inv_ornt_aff(trn, (1,1,1))
    revaff1i = nibabel.orientations.inv_ornt_aff(trn_back, (1,1,1))
    revaff64i = nibabel.orientations.inv_ornt_aff(trn_back, (64,64,64))
    aff_reor64 = np.linalg.lstsq(
        bbox_world(revaff64i, (64,64,64)), 
        bbox_world(img.affine, img.shape[:3]), 
        rcond=None)[0].T
    
    wgridt = (netAff.grid @ torch.tensor(revaff1i, device=device, dtype=torch.float32))[None,...,[2,1,0]]
    d_orr = F.grid_sample(
        torch.as_tensor(d, dtype=torch.float32, device=device)[None,None], 
        wgridt, align_corners=True)
    
    # Head segmentation
    with torch.no_grad():
        out1t = net(d_orr)
    out1 = np.asarray(out1t.cpu())
    
    # Brain mask
    brainmask_cc = torch.tensor(out1[0,0].astype("float32"))
    
    # MNI registration
    with torch.no_grad():
        wc1, tA = netAff(out1t[:,[1,3]] * brainmask_cc)
    
    wnat = np.linalg.lstsq(
        bbox_world(img.affine, img.shape[:3]), 
        bbox_one @ revaff1, rcond=None)[0]
    wmni = np.linalg.lstsq(
        bbox_world(affine64_mni, (64,64,64)), 
        bbox_one, rcond=None)[0]
    M = (wnat @ inv(np.asarray(tA[0].cpu())) @ inv(wmni)).T
    
    # Hippocampus ROI
    imgcroproi_affine = np.array([[-1., 0., 0., 54.], 
                                   [0., 1., 0., -59.], 
                                   [0., 0., 1., -45.], 
                                   [0., 0., 0., 1.]])
    imgcroproi_shape = (107, 72, 68)
    
    gsx, gsy, gsz = imgcroproi_shape
    sgrid = np.rollaxis(indices_unitary((gsx,gsy,gsz), dtype=np.float32), 0, 4)
    
    bboxnat = bbox_world(imgcroproi_affine, imgcroproi_shape) @ inv(M.T) @ wnat
    matzoom = np.linalg.lstsq(bbox_one, bboxnat, rcond=None)[0]
    wgridt = torch.tensor(
        mul_homo(sgrid, (matzoom @ revaff1i))[None,...,[2,1,0]], 
        device=device, dtype=torch.float32)
    dout = F.grid_sample(
        torch.as_tensor(d, dtype=torch.float32, device=device)[None,None], 
        wgridt, align_corners=True)
    d_in = np.asarray(dout[0,0].cpu())
    
    d_in -= d_in.mean()
    d_in /= d_in.std()
    
    # Segment hippocampus
    with torch.no_grad():
        hippoR = hipponet(torch.as_tensor(d_in[None, None, 6:54:+1, :, 2:-2].copy()))
        hippoL = hipponet(torch.as_tensor(d_in[None, None, -7:-55:-1, :, 2:-2].copy()))
    
    hippoRL = np.vstack([np.asarray(hippoR.cpu()), np.asarray(hippoL.cpu())])
    hippoRL = np.clip(((hippoRL - .5) * 2 + .5), 0, 1) * (hippoRL > .5)
    
    output = np.zeros((2, 107, 72, 68), np.float32)
    output[0, -7:-55:-1, :, 2:-2][2:-2, 2:-2, 2:-2] = np.clip(hippoRL[1] * 255, 0, 255)
    output[1, 6:54:+1, :, 2:-2][2:-2, 2:-2, 2:-2] = np.clip(hippoRL[0] * 255, 0, 255)
    
    # Resample to native space
    pts = bbox_xyz(imgcroproi_shape, imgcroproi_affine)
    pts = mul_homo(pts, np.linalg.inv(M).T)
    pts_ijk = mul_homo(pts, np.linalg.inv(img.affine).T)
    for i in range(3):
        np.clip(pts_ijk[:,i], 0, img.shape[i], out=pts_ijk[:,i])
    pmin = np.floor(np.min(pts_ijk, 0)).astype(int)
    pwidth = np.ceil(np.max(pts_ijk, 0)).astype(int) - pmin
    
    widx = indices_xyz(pwidth, img.affine, offset_vox=pmin)
    widx = mul_homo(widx, M.T)
    DHW3 = xyz_to_DHW3(widx, imgcroproi_affine, imgcroproi_shape)
    
    wdata = np.zeros(img.shape[:3], np.uint8)
    
    # Left hippocampus
    d = torch.tensor(output[0].T, dtype=torch.float32)
    outDHW = F.grid_sample(d[None,None], torch.tensor(DHW3[None]), align_corners=True)
    dnat = np.asarray(outDHW[0,0].permute(2,1,0))
    dnat[dnat < 32] = 0
    wdata[pmin[0]:pmin[0]+pwidth[0], pmin[1]:pmin[1]+pwidth[1], pmin[2]:pmin[2]+pwidth[2]] = dnat.astype(np.uint8)
    L_output = wdata.copy()
    
    # Right hippocampus
    wdata = np.zeros(img.shape[:3], np.uint8)
    d = torch.tensor(output[1].T, dtype=torch.float32)
    outDHW = F.grid_sample(d[None,None], torch.tensor(DHW3[None]), align_corners=True)
    dnat = np.asarray(outDHW[0,0].permute(2,1,0))
    dnat[dnat < 32] = 0
    wdata[pmin[0]:pmin[0]+pwidth[0], pmin[1]:pmin[1]+pwidth[1], pmin[2]:pmin[2]+pwidth[2]] = dnat.astype(np.uint8)
    R_output = wdata.copy()

    return nibabel.Nifti1Image(L_output, img.affine), nibabel.Nifti1Image(R_output, img.affine)


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python segmentation.py <input_t1_image.nii.gz>")
        sys.exit(1)

    filename = sys.argv[1]
    L_img, R_img = segment_hippocampus(filename)

    base = filename[:-7] if filename.endswith(".nii.gz") else os.path.splitext(filename)[0]
    L_path = base + "_hippo_L.nii.gz"
    R_path = base + "_hippo_R.nii.gz"
    nibabel.save(L_img, L_path)
    nibabel.save(R_img, R_path)

    L_arr = np.asanyarray(L_img.dataobj)
    R_arr = np.asanyarray(R_img.dataobj)
    print(f"\nSegmentation complete!")
    print(f"Left hippocampus:  {L_img.shape}, non-zero voxels: {np.count_nonzero(L_arr)} -> {L_path}")
    print(f"Right hippocampus: {R_img.shape}, non-zero voxels: {np.count_nonzero(R_arr)} -> {R_path}")
