import streamlit as st
import py3Dmol
from stmol import showmol
import PIL.Image
import xmlrpc.client as xlmrpclib
from PIL import ImageColor
import ast
import os
import time
import tempfile
import time
import cv2
from collections import Counter
from sklearn.cluster import KMeans
from matplotlib import colors
import matplotlib.pyplot as plt
import numpy as np
import json
from io import BytesIO


st.set_option("deprecation.showfileUploaderEncoding", False)
st.title("2D and 3D Molecular Structure Stylization using ML")
cmd = xlmrpclib.ServerProxy("http://localhost:9123/")
st.sidebar.title("Select the mode")
# select 2D/3D stylization
tool = st.sidebar.selectbox(
    label="",
    options=[
        "2D",
        "3D",
    ],
)
if tool == "2D":
    protein = st.text_input("PDB Entry")
    bcolor = st.color_picker("Pick A Color", "#00f900")
    rgb = ImageColor.getcolor(bcolor, "RGB")
    xyzview = py3Dmol.view(query="pdb:" + protein)
    xyzview.setStyle({"sphere": {"color": bcolor}})
    xyzview.addSurface(py3Dmol.SAS, {"opacity": 1, "color": bcolor})
    xyzview.setBackgroundColor("white")
    xyzview.zoomTo()
    # show the initial protein
    if st.button("Show"):
        showmol(xyzview, height=500, width=800)
        cmd.fetch(protein)
        cmd.do(
            f"""     
                bg_color black
                as surface              
                set_color prot, {rgb}
                color prot
                "ray_trace_color, black
                ray 500, 500
                """
        )
        cmd.png("./2d_cyclegan/datasets/testA/initial.png")
        time.sleep(1)
        start_time = time.time()
        # make stylization
        os.system(
            "python3 ./2d_cyclegan/test_gan.py \
--dataroot ./2d_cyclegan/datasets/testA \
--name pdb2good \
--model test \
--no_dropout \
--model_suffix _A \
--preprocess none"
        )
        # FPS result
        print("FPS: ", 1 / (time.time() - start_time))
        image = PIL.Image.open("./results/pdb2good/test_latest/images/initial_fake.png")
        st.title("Goodsell-like Stylized Result")
        # show the stylized 2D result
        st.image(image)
        # an option to download the stylized result
        buf = BytesIO()
        image.save(buf, format="JPEG")
        byte_im = buf.getvalue()
        btn = st.download_button(
            label="Download Stylized Protein",
            data=byte_im,
            file_name="imagename.png",
            mime="image/jpeg",
        )
if tool == "3D":
    protein = st.text_input("PDB Entry")
    bcolor = st.color_picker("Pick A Color", "#00f900")
    rgb = ImageColor.getcolor(bcolor, "RGB")
    xyzview = py3Dmol.view(query="pdb:" + protein)
    xyzview.setStyle({"sphere": {"color": bcolor}})
    xyzview.addSurface(py3Dmol.SAS, {"opacity": 1, "color": bcolor})
    xyzview.setBackgroundColor("white")
    xyzview.zoomTo()
    st.title("Initial View")
    # show the initial protein
    showmol(xyzview, height=500, width=800)
    # choose the style for transferring
    if st.button("Goodsell-like"):
        uploaded_file = st.file_uploader(
            "Upload Style Goodsell-like Image",
            accept_multiple_files=False,
        )
        # show the style image
        if uploaded_file is not None:
            st.title("Style Image")
            image = PIL.Image.open(uploaded_file)
            st.image(image)
            # set params for further optimization
            params = {
                "ambient": 0,
                "ray_trace_gain": 0,
                "ray_trace_mode": 3,
                "cull_spheres": -1,
                "sphere_scale": 1,
                "sphere_transparency": 0,
                "sphere_mode": -1,
                "sphere_solvent": 0,
            }
            # write in these parameters
            with open("params.txt", "w") as file:
                file.write(json.dumps(params))
            # run stylization in PyMOL
            os.system(
                f"python3 ./3D_PRoteins_Params/src/solver.py  --protein {protein} --representation spheres  --style_image_path {uploaded_file} --params_txt params.txt --compare_method neural"
            )
            # read the best resulted parameters
            file = open("texture_solver_params.txt", "r")
            cont = file.read()
            result_params = ast.literal_eval(cont)
            cmd.fetch(protein)
            # stylize with resulted parameters
            cmd.do(
                f"""
                        as spheres
                        """
            )
            for k, v in result_params.items():
                # transfer colors of the style image
                if k == "color_1":
                    cmd.do(
                        f""" set_color basic, {v}
                            color basic, org
                        """
                    )
                elif k == "color_2":
                    cmd.do(
                        f""" set_color ligg, {v}
                            color ligg, lig
                        """
                    )
                # transfer other parameters
                else:
                    cmd.do(f"""set {k}, {v}""")

    # another 3D style choice
    if st.button("Geis-like"):
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Sticks"):
                uploaded_file = st.file_uploader(
                    "Upload Style Geis-like Image",
                    accept_multiple_files=False,
                )
                if uploaded_file is not None:
                    st.title("Style Image")
                    image = PIL.Image.open(uploaded_file)
                    # show Geis style image
                    st.image(image)
                    params = {
                        "stick_ball": 1,
                        "stick_overlap": 0,
                        "stick_radius": 0.05,
                        "stick_ball_ratio": 7,
                        "stick_radius": 0.25,
                        "stick_fixed_radius": 0,
                        "stick_nub": 0.7,
                        "stick_transparency": 0,
                        "stick_ball": 0,
                        "stick_color": -1,
                        "stick_overlap": 0.2,
                        "stick_quality": 8,
                        "stick_valence_scale ": 1,
                    }
                    # write in initial parameters for further optimization
                    with open("params.txt", "w") as file:
                        file.write(json.dumps(params))
                    # run stylization
                    os.system(
                        f"python3 ./3D_PRoteins_Params/src/solver.py  --protein {protein} --representation sticks  --style_image_path {uploaded_file} --params_txt params.txt --compare_method neural"
                    )
                    # read the best resulted parameters
                    file = open("texture_solver_params.txt", "r")
                    cont = file.read()
                    result_params = ast.literal_eval(cont)
                    cmd.fetch(protein)
                    # stylize with resulted parameters
                    cmd.do(
                        f"""
                        as sticks
                        """
                    )
                    for k, v in result_params.items():
                        # transfer colors of the style image
                        if k == "color_1":
                            cmd.do(
                                f""" set_color basic, {v}
                            color basic, org
                        """
                            )
                        elif k == "color_2":
                            cmd.do(
                                f""" set_color ligg, {v}
                            color ligg, lig
                        """
                            )
                        # transfer other parameters
                        else:
                            cmd.do(f"""set {k}, {v}""")
        with col2:
            if st.button("Cartoon"):
                uploaded_file = st.file_uploader(
                    "Upload Style Goodsell-like Image",
                    accept_multiple_files=False,
                )
                if uploaded_file is not None:
                    st.title("Style Image")
                    image = PIL.Image.open(uploaded_file)
                    st.image(image)
                    params = {
                        "surface_mode": 3,
                        "transperency_mode": 1,
                        "transparency": 0.7,
                        "ray_transparency_oblique": 1,
                        "ray_transparency_oblique_power": 8,
                        "cartoon_cylindrical_helices": 0,
                        "cartoon_debug": 0,
                        "cartoon_dumbbell_length": 1.6,
                        "cartoon_dumbbell_radius": 0.16,
                        "cartoon_dumbbell_width": 0.17,
                        "cartoon_fancy_helices": 0,
                        "cartoon_fancy_sheets": 0,
                        "cartoon_flat_sheets": 1,
                        "cartoon_loop_cap": 1,
                        "cartoon_nucleic_acid_mode": 4,
                        "cartoon_oval_quality": 10,
                        "cartoon_ring_finder": 1,
                        "cartoon_smooth_cycles": 2,
                        "cartoon_transparency": 0,
                        "cartoon_tube_cap": 2,
                    }
                    with open("params.txt", "w") as file:
                        file.write(json.dumps(params))
                    os.system(
                        f"python3 ./3D_PRoteins_Params/src/solver.py  --protein {protein} --representation cartoon --cartoon_geis_mode True  --style_image_path {uploaded_file} --params_txt params.txt --compare_method neural"
                    )

                    # read the best resulted parameters
                    file = open("texture_solver_params.txt", "r")
                    cont = file.read()
                    result_params = ast.literal_eval(cont)
                    cmd.fetch(protein)
                    # stylize with resulted parameters
                    cmd.do(
                        f"""
                        as cartoon
                        """
                    )
                    for k, v in result_params.items():
                        # transfer colors of the style image
                        if k == "color_1":
                            cmd.do(
                                f""" set_color basic, {v}
                            color basic
                        """
                            )
                        elif k == "color_2":
                            cmd.do(
                                f""" set_color bg, {v}
                                    bg_color, bg
                        """
                            )
                        # transfer other parameters
                        else:
                            cmd.do(f"""set {k}, {v}""")
