# COPICK: An open-source adaptation of Opentrons OT-2 robot to support colony picking  
Framework designed to adapt Opentrons 2 robot for colony picking protocols.

Colony picking is a labware based protocol involving the operation of several pieces of hardware in a unique sequence of tasks. The core of such workflow is the visual identification of colonies on the surface of agar plates. Although OT-2 is an open source platform exhibiting great operational capabilities, it lacks of any kind of computer vision recognition system. As a consequence, OT-2 cannot support colony picking.
COPICK package offers a technical solution to enable OT-2 as a colony picking platform at hardware and software level, spotting the two main problems in such task:

-	Computer vision of agar plates is performed using a convolutional neural network with an architecture proposed by Facebook research team (Detectron2, https://github.com/facebookresearch/detectron2), specifically trained with a “de novo” colony plate database. This database contains images of petri dishes with both M9 and LB agar containing P.putida and E.coli colonies.

-	The framework assembly to adapt OT-2 robot for colony picking operations. It includes CAD models for a custom-made scaffolding to assemble a DIY transilluminator (based on a cheap RGB LED matrix powered by a Raspberry Pi) and a high sensitivity USB color camera. The whole setup is complemented with all the scripts required to synchronize the image computation with robot movement and light activation, convert data gathered from the image to actual picking process and use typical filter criterion of colonies (size, color, GFP).

Information about both implementations can be found inside each respective folder.

# Citation

This project is also presented and explained in a formal published manuscript in BioArxiv: XXXXX

If you want to cite this work, use the following reference:
xxxx
xxxx
xxxx

# Contact

Questions and suggestions are welcome. Feel free to contact us on david.rodriguez@cnb.csic.es



