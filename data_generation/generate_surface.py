import biotite.database.rcsb as rcsb
import xmlrpc.client as xlmrpclib
import tqdm
import argparse

"""
Before this download and launch pymol: "pymol -R"
This function generates surface representations (.png) of the proteins from PDB, which resolution is less_or_equeal certain number. Recommended -- 1/1.5/2. This will generate 1249/19124/77027 images accordingly.

"""

def generate(resolution):
	query = rcsb.FieldQuery("reflns.d_resolution_high", less_or_equal= resolution)
	pdbs = sorted(rcsb.search(query))
	cmd = xlmrpclib.ServerProxy("http://localhost:9123/")
	for pdb_id in tqdm(pdbs):
    		cmd.delete('all')
    		cmd.orient()
    		cmd.fetch(pdb_id)
    	cmd.show_as('surface', pdb_id)
    	cmd.color('gray')
    	cmd.ray()
    	cmd.set('ray_opaque_background',  'off')
    	cmd.png(f'./{pdb_id}.png')


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
    	parser.add_argument('--resolution', dest='resolution', type=int)
	generate(resolution)