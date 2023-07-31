from tdw.asset_bundle_creator import AssetBundleCreator
import os

for i in range(6):
    for j in range(6):
        id = str(i)+str(j)
        a = AssetBundleCreator()
        # Change this to the actual path.
        model_path = os.path.abspath("/landscape/landV3.1/slice{}.obj".format(id))
        
        asset_bundle_paths, record_path = a.create_asset_bundle(model_path, True, 123, "", 1)
