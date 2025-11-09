![logo](https://imaging.epfl.ch/resources/logo-for-gitlab.svg)
![epfl](./assets/epfl.jpeg)
# Awesome Scientific Image Analysis [![Awesome](https://awesome.re/badge.svg)](https://awesome.re)

A curated list of scientific image analysis resources and software tools.

## 📌 Contents

- [🔖 Getting started](#-getting-started)
- [🧑‍🤝‍🧑 Communities](#-communities)
- [📚 Learning resources](#-learning-resources)
- [✂️ Image segmentation](#️-image-segmentation)
- [📐 Image registration](#-image-registration)
- [🪄 Image denoising](#-image-denoising)
- [🔍 Object detection](#-object-detection)
- [🐾 Tracking](#-tracking)
- [🌻 Visualization](#-visualization)
- [🔋 Performance](#-performance)
- [🕊️ Open science](#️-open-science)
- [🐍 Python](#-python)
- [🔬 Fiji (ImageJ)](#-fiji-imagej)
- [🏝️ Napari](#️-napari)
- [🧬 QuPath](#-qupath)
- [🏗️ Infrastructure](#️-infrastructure)
- [🛸 Other](#-other)

## 🔖 Getting started

Online courses to learn scientific image analysis:

- [Image Processing and Analysis for Life Scientists](https://courseware.epfl.ch/courses/course-v1:EPFL+IPA4LS+2019_t3/about) - BIOP, EPFL.
- [Introduction to Bioimage Analysis](https://bioimagebook.github.io/README.html) - Pete Bankheads.
- [Image Processing with Python](https://datacarpentry.github.io/image-processing/) - Data Carpentry.
- [Image data science with Python and Napari](https://biapol.github.io/Image-data-science-with-Python-and-Napari-EPFL2022/intro.html) - EPFL & TU Dresden.
- [bioimagingguide.org](https://www.bioimagingguide.org/welcome.html) - Center for Open Bioimage Analysis.

Courses in video format:

- [First principles in computer vision](https://www.youtube.com/channel/UCf0WB91t8Ky6AuYcQV0CcLw) - Columbia University.
- [Introduction to bioimage analysis](https://www.youtube.com/watch?v=e-2DbkUwKk4&list=PL5ESQNfM5lc7SAMstEu082ivW4BDMvd0U&index=3) - Robert Haase.
- [Microscopy Series](https://www.ibiology.org/online-biology-courses/microscopy-series/) - iBiology. Focused on microscopy techniques.

General image analysis software:

- [Fiji](https://fiji.sc/) - ImageJ, with “batteries-included”.
- [Ilastik](https://www.ilastik.org/) - Interactive learning and segmentation toolkit.
- [Napari](https://napari.org/) - A fast and interactive multi-dimensional image viewer for Python.
- [QuPath](https://qupath.github.io/) - Open Software for Bioimage Analysis.

Python:

- [Setting up Python for scientific image analysis](https://imaging.epfl.ch/field-guide/sections/python/notebooks/python_setup.html) - Short guide by the EPFL Center for Imaging.
- [Introduction to Python for Image Analysis](https://epfl-center-for-imaging.github.io/python-intro-images/lab/index.html?path=python-intro%2FContent.ipynb) - Jupyterlite Notebook (no installation required).

## 🧑‍🤝‍🧑 Communities

- [Image.sc](https://image.sc/) - Popular online forum focused on bioimage analysis.
- [GloBIAS](https://www.globias.org/) - Global Bioimage Analysts' Society.
<!-- - [SwissBIAS](https://swissbias.github.io/) -->
<!-- - [Euro-BioImaging](https://www.eurobioimaging.eu/) -->
<!-- - [QUAREP-LiMi](https://quarep.org/) -->
<!-- - [Smart Microscopy](https://smartmicroscopy.org/) -->
<!-- - [BIII.eu](https://biii.eu/) -->

## 📚 Learning resources

### Curated lists

- [Awesome Biological Image Analysis](https://github.com/hallvaaw/awesome-biological-image-analysis)
- [Awesome Computer Vision](https://github.com/jbhuang0604/awesome-computer-vision)
- [Awesome Medical Imaging](https://github.com/fepegar/awesome-medical-imaging)

### Papers

- [2024 - Creating and troubleshooting microscopy analysis workflows: Common challenges and common solutions](https://onlinelibrary.wiley.com/doi/full/10.1111/jmi.13288) - Beth Cimini.
- [2023 - Towards effective adoption of novel image analysis methods](https://www.nature.com/articles/s41592-023-01910-2) - Talley Lambert, Jennifer Waters.
- [2022 - A Hitchhiker's guide through the bio-image analysis software universe](https://febs.onlinelibrary.wiley.com/doi/full/10.1002/1873-3468.14451) - Robert Haase et al.

### Video series

- [DigitalSreeni](https://www.youtube.com/c/DigitalSreeni) - Focused on Python and deep learning for image analysis.
<!-- - [Microcourses](https://www.youtube.com/@Microcourses/videos) -->
<!-- - [Optical microscopy Image Processing & analysis](https://www.youtube.com/@johanna.m.dela-cruz/videos) -->
<!-- - [Aits Lab](https://www.youtube.com/channel/UCmh81PBL4lU6r6mcGqhRPbQ/playlists) -->

<!-- **Blogs** -->
<!-- - [Did you know – Image Analysis Style](https://didyouknowimageanalysis.wordpress.com/) - Marie Held -->

## ✂️ Image segmentation

Image segmentation aims to create a segmentation mask that identifies specific classes or objects. Techniques for image segmentation include thresholding, weakly supervised learning (e.g., Ilastik, Weka), and deep learning.

### Learning resources

- [Thresholding - Introduction to Bioimage Analysis](https://bioimagebook.github.io/chapters/2-processing/3-thresholding/thresholding.html)
- [Image segmentation - Image data science with Python and Napari](https://biapol.github.io/Image-data-science-with-Python-and-Napari-EPFL2022/day2d_image_segmentation/readme.html)
<!-- - [Segmentation - ImageJ Tutorials](https://imagej.net/imaging/segmentation) -->
<!-- - [Thresholding - Scikit-image](https://scikit-image.org/docs/dev/auto_examples/segmentation/plot_thresholding.html) -->

### Software tools

- [skimage.segmentation](https://scikit-image.org/docs/stable/api/skimage.segmentation.html) - Classical segmentation algorithms in Python.
- [Ilastik - Pixel Classification](https://www.ilastik.org/documentation/pixelclassification/pixelclassification) - Semi-supervised workflow for pixel classification.
- [Segment Anything Model 2 (SAM 2)](https://github.com/facebookresearch/sam2) - Promptable, foundation model for image segmentation.
- [SAMJ](https://github.com/segment-anything-models-java/SAMJ-IJ) - Segment Anything in Fiji.
- [YOLO11 - Instance Segmentation](https://docs.ultralytics.com/tasks/segment/) - Image segmentation using Ultralytics YOLO.
- [rembg](https://github.com/danielgatis/rembg) - Remove image backgrounds.
- [nnUNet](https://github.com/MIC-DKFZ/nnUNet) - U-Net based biomedical image segmentation (2D and 3D).
- [segmentation_models](https://github.com/qubvel/segmentation_models) - Segmentation models with pretrained backbones in Keras (TensorFlow).
- [segmentation_models.pytorch](https://github.com/qubvel/segmentation_models.pytorch) - Segmentation models with pretrained backbones in PyTorch.
- [Monai](https://github.com/Project-MONAI/MONAI) - Pytorch-based deep learning framework for biomedical imaging.
- [StarDist](https://github.com/stardist/stardist) - Segmentation of cell nuclei and other round (star-convex) objects.
- [CellPose](https://github.com/mouseland/cellpose) - Segmentation of cells and membranes in microscopy images.
<!-- - [Detectron2](https://github.com/facebookresearch/detectron2) -->
<!-- - [pytorch-3dunet](https://github.com/wolny/pytorch-3dunet) -->
<!-- - [InstanSeg](https://github.com/instanseg/instanseg/tree/main) -->
<!-- - [omnipose](https://github.com/kevinjohncutler/omnipose) -->

## 📐 Image registration

Image registration is used to align multiple images, stabilize sequences by compensating for camera movement, track object movement and deformation, and stitch multiple fields of view together.

### Learning resources

- [Image correlation - Practice](https://www.spam-project.dev/docs/tutorials/tutorial-02b-DIC-practice.html)
- [Image correlation - Theory](https://www.spam-project.dev/docs/tutorials/tutorial-02a-DIC-theory.html)
- [Intro to Image Registration](https://www.youtube.com/watch?v=zDaCVSXMIm4) - Ella Bahry.

### Software tools

- [skimage.registration](https://scikit-image.org/docs/stable/api/skimage.registration.html) - Cross-correlation and optical flow algorithms in Python.
- [SPAM](https://www.spam-project.dev/) - Image correlation in 2D and 3D.
- [pystackreg](https://github.com/glichtner/pystackreg) - Image stack (or movie) alignment in Python.
- [TurboReg](https://bigwww.epfl.ch/thevenaz/turboreg/) - Image stack (or movie) alignment in Fiji.
- [Warpy](https://imagej.net/plugins/bdv/warpy/warpy) - Register whole slide images in Fiji.
- [ABBA](https://github.com/BIOP/ijp-imagetoatlas) - Aligning Big Brains and Atlases.
- [Fast4DReg](https://imagej.net/plugins/fast4dreg) - 3D drift correction in Fiji.
<!-- - [pyGPUreg](https://github.com/bionanopatterning/pyGPUreg) -->
<!-- - [Fast4DReg](https://imagej.net/plugins/fast4dreg) -->
<!-- - [SimpleElastix](https://simpleelastix.readthedocs.io/index.html) -->
<!-- - [DIPY](https://github.com/dipy/dipy) -->
<!-- - [ANTsPy](https://github.com/ANTsX/ANTsPy) -->
<!-- - [VoxelMorph](https://github.com/voxelmorph/voxelmorph) -->
<!-- https://github.com/NHPatterson/wsireg -->

## 🪄 Image denoising

Image denoising enhances visual quality by removing noise, making structures more distinguishable and facilitating segmentation through thresholding.

### Learning resources

- [Noise - Introduction to Bioimage Analysis](https://bioimagebook.github.io/chapters/3-fluorescence/3-formation_noise/formation_noise.html)
- [Denoising a picture (scikit-image)](https://scikit-image.org/docs/stable/auto_examples/filters/plot_denoise.html)

### Software tools

- [skimage.restoration](https://scikit-image.org/docs/stable/api/skimage.restoration.html) - Classical denoising algorithms in Python (TV Chambolle, Non-local Means, etc.).
- [CAREamics](https://github.com/CAREamics/careamics) - Deep-learning based, self-supervised algorithms: Noise2Void, N2V2, etc. 
- [CSBDeep](https://imagej.net/plugins/csbdeep) - Image restoration in Fiji.
- [noise2self](https://github.com/czbiohub-sf/noise2self) - Blind denoising with self-supervision.
- [CellPose3 - OneClick](https://cellpose.readthedocs.io/en/latest/restore.html) - Deep-learning based denoising models for fluorescence and microscopy images.
- [SwinIR](https://github.com/JingyunLiang/SwinIR/releases) - Deep image restoration using Swin Transformer - for grayscale and color images.

## 🔍 Object detection

Object detection is the process of identifying and localizing objects within an image or video using various shapes such as bounding boxes, keypoints, circles, or other geometric representations.

### Software tools

- [YOLO11 - Object Detection](https://github.com/ultralytics/ultralytics) - Object detection using Ultralytics YOLO.
- [DeepLabCut](https://www.mackenziemathislab.org/deeplabcut) - Animal pose estimation.
- [OpenPifPaf](https://github.com/openpifpaf/openpifpaf) - Human pose estimation.
- [Spotiflow](https://github.com/weigertlab/spotiflow) - Spot detection for microscopy data.
<!-- - [Detectron2](https://github.com/facebookresearch/detectron2) -->

## 🐾 Tracking

Object tracking is the process of following objects across time in a video or image time series.

### Learning resources

- [Getting started with TrackMate](https://imagej.net/plugins/trackmate/tutorials/getting-started)
- [Walkthrough (trackpy)](https://soft-matter.github.io/trackpy/dev/tutorial/walkthrough.html)
- [Single cell tracking with napari](https://napari.org/stable/tutorials/tracking/cell_tracking.html)
<!-- - [Trackmate Introduction and Demo](https://www.youtube.com/watch?v=7HWtaikIFcs) -->

### Software tools

- [TrackMate](https://imagej.net/plugins/trackmate/) - Fiji plugin.
- [Trackpy](https://github.com/soft-matter/trackpy) - Particle tracking in Python.
- [Trackastra](https://github.com/weigertlab/trackastra) - Tracking with Transformers.
- [ultrack](https://github.com/royerlab/ultrack) - Large-scale cell tracking.
- [co-tracker](https://github.com/facebookresearch/co-tracker) - Tracking any point on a video.
- [LapTrack](https://github.com/yfukai/laptrack) - Particle tracking in Python.
- [Mastodon](https://imagej.net/plugins/mastodon) - Large-scale tracking in Fiji.
- [SAM-PT](https://github.com/SysCV/sam-pt) - Segment Anything Meets Point Tracking.
<!-- - [DeepTrack2](https://github.com/DeepTrackAI/DeepTrack2) - Deep learning framework for digital microscopy. -->

## 🌻 Visualization

A variety of software tools are available for visualizing scientific images and their associated data.

For a detailed comparison of 3D viewers, see *[3D Image Visualization software tools](./extra/visualization.md)*.

### Learning resources

- [Visual image comparison (Scikit-image)](https://scikit-image.org/docs/stable/auto_examples/applications/plot_image_comparison.html#sphx-glr-auto-examples-applications-plot-image-comparison-py)

### Software tools

- [Fiji - Volume Viewer](https://imagej.net/plugins/volume-viewer) - Ideal for Fiji users.
- [Fiji - 3D Viewer](https://imagej.net/plugins/3d-viewer/) - Ideal for Fiji users.
- [Fiji - 3Dscript](https://imagej.net/plugins/3dscript) - 3D rendering animations in Fiji.
- [Napari](https://napari.org/stable/) - Interactive nD image viewer in Python.
- [PyVista](https://pyvista.org/) - 3D visualizations in Python through VTK.
- [vedo](https://github.com/marcomusy/vedo) - Scientific visualizations of 3D objects.
- [itkwidgets](https://github.com/InsightSoftwareConsortium/itkwidgets) - VTK viewer in Jupyter notebooks.
- [stackview](https://github.com/haesleinhuepf/stackview/) - 3D stack visualization in Jupyter notebooks.
- [Paraview](https://www.paraview.org/) - Scientific visualizations through VTK.
- [tif2blender](https://github.com/oanegros/tif2blender) - Microscopy image visualization in Blender.
- [Fiji - BigDataViewer](https://imagej.net/plugins/bdv/) - Ideal for big data.
- [Neuroglancer](https://github.com/google/neuroglancer) - Browser-based visualizations compatible with large images (zarr).
- [vizarr](https://github.com/hms-dbmi/vizarr) - Simple Zarr viewer.
- [Viv](https://github.com/hms-dbmi/viv?tab=readme-ov-file) - Multiscale visualization on the web.
<!-- - [K3D-jupyter](https://k3d-jupyter.org/index.html) -->
<!-- - [fastplotlib](https://github.com/fastplotlib/fastplotlib) -->
<!-- - [microfilm](https://github.com/guiwitz/microfilm) -->
<!-- - [microviewer](https://github.com/seung-lab/microviewer) -->
<!-- - [hyperspy](https://github.com/hyperspy/hyperspy) -->
<!-- - [ndv](https://github.com/pyapp-kit/ndv) -->
<!-- - [supervision](https://github.com/roboflow/supervision) -->
<!-- - [NeuroMorph](https://github.com/NeuroMorph-EPFL/NeuroMorph) -->
<!-- - [3D Slicer](https://www.slicer.org/) -->

## 🔋 Performance

Performance optimization is the process of making code execution faster, more efficient, or using fewer computing resources.

### Learning resources

- [System aspects - Basics of Computing Environments for Scientists](https://compenv.phys.ethz.ch/system_aspects/)
- [Accelerated large-scale image procesing in Python](https://github.com/EPFL-Center-for-Imaging/accel-large-image-proc-talk)

### Software tools

- [pyclesperanto_prototype](https://github.com/clEsperanto/pyclesperanto_prototype) - GPU-accelerated bioimage analysis.
- [Numba](https://numba.pydata.org/) - JIT compiler for Python and NumPy code.
- [cuCIM](https://github.com/rapidsai/cucim) - GPU-accelerated image processing.
- [OpenCV](https://opencv.org/) - Optimized image processing algorithms.

## 🕊️ Open science

Open imaging science meets principles of findability, accessibility, interoperability, and reusability (FAIR).

### Software development practices

- [The Turing Way handbook](https://the-turing-way.netlify.app/index.html)
- [Code Publishing cheat sheet](https://www.epfl.ch/schools/enac/wp-content/uploads/2022/06/ENAC-IT4R_Code_Publishing_Cheat_Sheet.pdf)
- [Good enough practices in scientific computing](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1005510)

### Reproducibility

- [Reproducible image handling and analysis](https://www.embopress.org/doi/full/10.15252/embj.2020105889)
- [Understanding metric-related pitfalls in image analysis validation](https://arxiv.org/abs/2302.01790)
- [Reporting reproducible imaging protocols](https://www.sciencedirect.com/science/article/pii/S2666166722009194?via%3Dihub)
- [When seeing is not believing: application-appropriate validation matters for quantitative bioimage analysis](https://www.nature.com/articles/s41592-023-01881-4)
<!-- - [Processing images for papers & posters](https://osf.io/a8hb6) -->

### Figures creation

- [Community-developed checklists for publishing images and image analysis](https://arxiv.org/abs/2302.07005)
- [Creating Clear and Informative Image-based Figures for Scientific Publications](https://www.biorxiv.org/content/10.1101/2020.10.08.327718v2)
- [Effective image visualization for publications – a workflow using open access tools and concepts](https://f1000research.com/articles/9-1373)
<!-- - [Publishing images for papers & posters](https://osf.io/mxhve) -->

## 🐍 Python

Python is a popular programming language for scientific image analysis.

### Python setup

- [Managing Conda Environments](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html)
- [Python environments workshop](https://hackmd.io/@talley/SJB_lObBi) - Talley Lambert.
- [Setting up Python for scientific image analysis](https://epfl-center-for-imaging.github.io/python-setup/) - EPFL Center for Imaging.
<!-- - [Conda Cheatsheet](https://docs.conda.io/projects/conda/en/latest/_downloads/843d9e0198f2a193a3484886fa28163c/conda-cheatsheet.pdf) -->

### Python programming

- [Python 3 documentation](https://docs.python.org/3/) - The official Python documentation.
- [Programming with Python](https://swcarpentry.github.io/python-novice-inflammation/index.html) - Software Carpentry.
<!-- - [Intermediate Research Software Development](https://carpentries-incubator.github.io/python-intermediate-development/) - Carpentries Incubator. -->
<!-- - [Python packaging 101](https://www.pyopensci.org/python-package-guide/tutorials/intro.html) -->
<!-- - [pydevtips: Python Development Tips](https://pydevtips.readthedocs.io/en/latest/index.html) - Eric Bezzam. -->
<!-- - [Automate the Boring Stuff with Python](https://automatetheboringstuff.com/) -->

### Python for image processing

- [Scikit-image](https://scikit-image.org/docs/stable/) - Scientific image processing toolbox.
- [scipy.ndimage](https://docs.scipy.org/doc/scipy/reference/ndimage.html) - Multidimensional image processing.
- [opencv-python](https://github.com/opencv/opencv-python) - Computer vision toolbox.
- [Introduction to Python for Image Analysis](https://epfl-center-for-imaging.github.io/python-intro-images/lab/index.html?path=Content.ipynb) - Notebook running on Jupyterlite. 
<!-- - [Image processing with Python](https://datacarpentry.org/image-processing/) - Data Carpentry. -->
<!-- - [Image processing with Python](https://github.com/guiwitz/Python_image_processing) - Guillaume Witz. -->
<!-- - [3.3. Scikit-image: image processing](https://scipy-lectures.org/packages/scikit-image/index.html) - Scientific Python Lectures. -->

## 🔬 Fiji (ImageJ)

Fiji is an open-source software for image processing and analysis. A wide range of community-developed plugins can enhance its functionality.

### Learning resources

- [Scientific Imaging Tutorials](https://imagej.net/imaging/index) - ImageJ.
- [Image handling using Fiji - training materials](https://zenodo.org/records/14771563) - Joanna Pylvänäinen.
<!-- - [Fiji Programming Tutorial](https://syn.mrc-lmb.cam.ac.uk/acardona/fiji-tutorial/) -->

### Plugins

- [MorphoLibJ](https://imagej.net/plugins/morpholibj) - Morphological operations.
- [DeepImageJ](https://deepimagej.github.io/) - Run deep learning models in Fiji.

## 🏝️ Napari

Napari is a fast and interactive multi-dimensional image viewer for Python. It can be used for browsing, annotating, and analyzing scientific images.

- [Usage (napari.org)](https://napari.org/stable/usage.html)

### Plugins

To browse all plugins, see [napari hub](https://www.napari-hub.org/).

- [napari-animation](https://github.com/napari/napari-animation) - Create animations.
- [napari-skimage-regionprops](https://github.com/haesleinhuepf/napari-skimage-regionprops) - Region properties.
- [napari-threedee](https://github.com/napari-threedee/napari-threedee) - 3D interactivity toolbox.
- [Omega](https://github.com/royerlab/napari-chatgpt) - Napari with ChatGPT.
- [napari-sam](https://github.com/MIC-DKFZ/napari-sam) - Segment Anything in Napari.
- [napari-imagej](https://github.com/imagej/napari-imagej) - Fiji in Napari.
- [devbio-napari](https://github.com/haesleinhuepf/devbio-napari) - Comprehensive image processing toolbox.
- [napari-clusters-plotter](https://github.com/BiAPoL/napari-clusters-plotter) - Object clustering.
- [napari-accelerated-pixel-and-object-classification](https://github.com/haesleinhuepf/napari-accelerated-pixel-and-object-classification) - Semi-supervised pixel classification.
- [napari-convpaint](https://github.com/guiwitz/napari-convpaint) - Pixel classification based on deep learning feature extraction.

## 🧬 QuPath

QuPath is an open software for bioimage analysis, often used to process and visualize digital pathology and whole slide images.

- [QuPath Documentation](https://qupath.readthedocs.io/en/stable/)

### Extensions

- [qupath-extension-cellpose](https://github.com/BIOP/qupath-extension-cellpose) - CellPose.
- [qupath-extension-stardist](https://github.com/qupath/qupath-extension-stardist) - StarDist.
- [qupath-extension-sam](https://github.com/ksugar/qupath-extension-sam) - Segment Anything in QuPath.

## 🏗️ Infrastructure

Infrastructure tools for image analysis workflows (and related).

- [BIOP-desktop](https://biop.github.io/biop-desktop-doc/) - Virtual desktop for bioimage analysis.
- [BAND](https://bandv1.denbi.uni-tuebingen.de/#/eosc-landingpage) - Bioimage ANalysis Desktop.
- [Fractal](https://fractal-analytics-platform.github.io/) - Framework to process bioimaging data at scale in the OME-Zarr format.
- [Galaxy (EU)](https://live.usegalaxy.eu/) - Web-based platform for accessible computational research.
- [Renkulab](https://renkulab.io/) - Data, Code, and Compute all under one roof.
- [Hugging Face Spaces](https://huggingface.co/spaces) - Build, host, and share ML apps.
- [BioImage.IO dev](https://dev.bioimage.io/) - Models, Datasets, and Applications for bioimage analysis.

## 🛸 Other

### 🤖 LLMs

- [bia-bob](https://github.com/haesleinhuepf/bia-bob)
- [BioImage.IO Chatbot](https://github.com/bioimage-io/bioimageio-chatbot)

### 📷 Image acquisition

- [Cameras and Lenses](https://ciechanow.ski/cameras-and-lenses/) - Bartosz Ciechanowski.
- [Knowledge Center](https://www.edmundoptics.eu/knowledge-center) - Edmund Optics.
<!-- - [Guides - Center for Microscopy and Image Analysis](https://zmb.dozuki.com/) - University of Zurich -->

### 🩻 Image reconstruction

- [Pyxu](https://pyxu-org.github.io/) - Modular and Scalable Computational Imaging.
<!-- - [Welcome to Inverse Problems and Imaging](https://tristanvanleeuwen.github.io/IP_and_Im_Lectures/intro.html) -->

### 🖌️ Image annotation

- [makesense.ai](https://www.makesense.ai/) - Simple annotation app for YOLO models.

### 💲 Splines

- [SplineBox](https://splinebox.readthedocs.io/en/latest/index.html) - Efficient splines fitting in Python.

### 🍭 Orientation

- [OrientationJ](https://bigwww.epfl.ch/demo/orientationj/) - Fiji plugin.
- [OrientationPy](https://epfl-center-for-imaging.gitlab.io/orientationpy/introduction.html) - 2D and 3D orientation measurements in Python.

## 🕸️ Meshes

- [scikit-shapes](https://github.com/scikit-shapes/scikit-shapes) - Shape processing in Python.

### 🛠️ Utilities

- [tifffile](https://pypi.org/project/tifffile/) - Read and write TIFF images.
- [aicsimageio](https://github.com/AllenCellModeling/aicsimageio) - Image reading and metadata conversion.
- [imageio](https://github.com/imageio/imageio) - Python library for reading and writing image data.
- [patchify](https://pypi.org/project/patchify/) - Image patching (tiling).
- [pims](https://soft-matter.github.io/pims/) - Python Image Sequence.
- [imutils](https://github.com/PyImageSearch/imutils) - Image utilities.
- [bioio](https://github.com/bioio-devs/bioio) - Read, write, and manage microscopy images.
<!-- - [imantics](https://imantics.readthedocs.io/en/latest/) - Image annotation semantics (Masks, Bounding Box, Polygons...) -->
<!-- - [ncolor](https://github.com/kevinjohncutler/ncolor) - Remapping of instance labels -->
<!-- - [pixelflow](https://github.com/alan-turing-institute/pixelflow) -->
