import ml_collections

def get_EM_2x3_2_hires_config():
    config = ml_collections.ConfigDict()
    config.if_transskip = True
    config.if_convskip = True
    config.patch_size = 2
    config.in_chans = 2
    config.embed_dim = 96 #8#96
    config.token_dim = 24 #token dim and embed dim reversed here
    config.depths = (2, 3)
    config.num_heads = (4, 8)
    config.axial = (('12','23'),('31','12','23'))
    config.window_size = (5, 6, 7)
    config.mlp_ratio = 4
    config.pat_merg_rf = 4
    config.qkv_bias = False
    config.drop_rate = 0
    config.drop_path_rate = 0.3
    config.ape = False
    config.spe = True
    config.rpe = False
    config.patch_norm = True
    config.use_checkpoint = False
    config.out_indices = (0, 1)
    config.reg_head_chan = 16
    config.img_size = (160, 192, 224)
    config.ds_dims = ((40,48,56),(20,24,28),(10,12,14),(5,6,7))
    return config

def get_EM_1x1_2_hires_config():
    config = ml_collections.ConfigDict()
    config.if_transskip = True
    config.if_convskip = True
    config.patch_size = 2
    config.in_chans = 2
    config.embed_dim = 96 #8#96
    config.token_dim = 24 #token dim and embed dim reversed here
    config.depths = (1, 1)
    config.num_heads = (4, 8)
    config.axial = (('12',),('23',),)
    config.mlp_ratio = 4
    config.pat_merg_rf = 4
    config.qkv_bias = False
    config.drop_rate = 0
    config.drop_path_rate = 0.3
    config.ape = False
    config.spe = True
    config.rpe = False
    config.patch_norm = True
    config.use_checkpoint = False
    config.out_indices = (0, 1)
    config.reg_head_chan = 16
    config.img_size = (160, 192, 224)
    config.ds_dims = ((40,48,56),(20,24,28),(10,12,14),(5,6,7))
    return config

def get_EM_1x1_2_config():
    config = ml_collections.ConfigDict()
    config.if_transskip = True
    config.if_convskip = True
    config.patch_size = 2
    config.in_chans = 2
    config.embed_dim = 24
    config.depths = (1, 1)
    config.num_heads = (4, 8)
    config.axial = (('12',),('23',),)
    config.mlp_ratio = 4
    config.pat_merg_rf = 4
    config.qkv_bias = False
    config.drop_rate = 0
    config.drop_path_rate = 0.3
    config.ape = False
    config.spe = False
    config.rpe = False
    config.patch_norm = True
    config.use_checkpoint = False
    config.out_indices = (0, 1)
    config.reg_head_chan = 16
    config.img_size = (160, 192, 224)
    config.ds_dims = ((40,48,56),(20,24,28),(10,12,14),(5,6,7))
    config.hires = True
    return config

def get_EM_2x3_2_config():
    config = ml_collections.ConfigDict()
    config.if_transskip = True
    config.if_convskip = True
    config.patch_size = 2
    config.in_chans = 2
    config.embed_dim = 24
    config.depths = (2, 3)
    config.num_heads = (4, 8)
    config.axial = (('12','23'),('31','12','23'))
    config.mlp_ratio = 4
    config.pat_merg_rf = 4
    config.qkv_bias = False
    config.drop_rate = 0
    config.drop_path_rate = 0.3
    config.ape = False
    config.spe = False
    config.rpe = False
    config.patch_norm = True
    config.use_checkpoint = False
    config.out_indices = (0, 1)
    config.reg_head_chan = 16
    config.img_size = (160, 192, 224)
    config.ds_dims = ((40,48,56),(20,24,28),(10,12,14),(5,6,7))
    return config


def get_EM_1x1_4_config():
    config = ml_collections.ConfigDict()
    config.if_transskip = True
    config.if_convskip = True
    config.patch_size = 4
    config.in_chans = 2
    config.embed_dim = 24
    config.depths = (1, 1)
    config.num_heads = (4, 8)
    config.axial = (('12',),('23',),)
    config.mlp_ratio = 4
    config.pat_merg_rf = 4
    config.qkv_bias = False
    config.drop_rate = 0
    config.drop_path_rate = 0.3
    config.ape = False
    config.spe = False
    config.rpe = False
    config.patch_norm = True
    config.use_checkpoint = False
    config.out_indices = (0, 1)
    config.reg_head_chan = 16
    config.img_size = (160, 192, 224)
    config.ds_dims = ((40,48,56),(20,24,28),(10,12,14),(5,6,7))
    return config

def get_EM_2x3_4_config():
    config = ml_collections.ConfigDict()
    config.if_transskip = True
    config.if_convskip = True
    config.patch_size = 4
    config.in_chans = 2
    config.embed_dim = 24
    config.depths = (2, 3)
    config.num_heads = (4, 8)
    config.axial = (('12','23'),('31','12','23'))
    config.mlp_ratio = 4
    config.pat_merg_rf = 4
    config.qkv_bias = False
    config.drop_rate = 0
    config.drop_path_rate = 0.3
    config.ape = False
    config.spe = False
    config.rpe = False
    config.patch_norm = True
    config.use_checkpoint = False
    config.out_indices = (0, 1)
    config.reg_head_chan = 16
    config.img_size = (160, 192, 224)
    config.ds_dims = ((40,48,56),(20,24,28),(10,12,14),(5,6,7))
    return config

