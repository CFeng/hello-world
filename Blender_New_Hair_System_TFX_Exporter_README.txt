This exporter is designed to add the functionality of exporting Blender new hair system models as TressFX (TFX) files. 
This is done by retrieving the information required by TFX asset loader from the Blender model using Blender Python API (Bpy).
The exporter makes sure that the number of vertices per strand is compatible with conditions set by TFX asset loader by either 
adding or removing vertices. Any type of change in the strands is reported to user with message boxes. One of the main features 
of the exporter is the subdivision algorithm which mimics the Blender subdivision algorithm for Catmull-Rom cuves.

How the Exporter Works
	The exporter extracts the control points of each strand and the value of additional subdivision chosen by user in Blender UI. 
	Based on these two values, the number of vertices in each strand after performing subdivision is calculated. This number might 
	not meet the conditions set by the TFX asset loader. In that case, the algorithm finds the closest larger number of vertices 
	acceptable by the asset loader and uses interpolation to add those vertices. If the number of vertices are higher than the maximum
	value expected by the asset loader, it first tries to decrease the number of vertices by applying fewer rounds of subdivision and
	if not possible, it applies vertex subsampling by removing some vertices. Every change made by the algorithm to the final exported
	hair strands are reported to the user with warning pop-up boxes. Both subdivision and subsampling methods are explained in the next
	sections. 
	
	- Subdivision Algorithm
		The subdivision algorithm (subdivide function in the code) mimics the subdivision algorithm for Catmull-Rom curves, with some 
		changes, implemented here under the interpolate_to_evaluated method in Blender source code. The algorithm goes over the strands 
		with a window of four points (called segments and treated as one curve) and moves the window further one point per iteration to 
		process the new segment. Note that for the first segment, the first vertex is included twice (v1, v1, v2, v3 if v is a vertex) 
		and for the last segment the last vertex is included twice (v_n-2, v_n-1, v_n, v_n). For each segment, the 4 vertices are replaced
		with a given number of vertices (by default 5 but can be a larger number if extra vertices are needed).

	- Subsample Algorithm
		The subsample algorithm, removes vertices by calculating a removal step with the goal of preserving the general strand shape in order
		to reduce the number of vertices to the maximum number acceptable by the asset loader.

Installing the Exporter
	Follow these steps to install the exporter in Blender:
		- Open Blender
		- At the top menu bar, select Edit => Prefrences
		- In Blender Preferences window, select Install
		- Find and select the exporter python script in the file browser window and hit Install Add-on.
		  by default the exporter should be located at {Engine Directory}/o3de/Gems/AtomTressFX/Tools/Blender/TressFX_Exporter_Blender.py
	The exporter should be installed now, note that you can enable and disable the add-ons using the Blender Prefrences menu.

Step by Step Example to Create a Hair Model Using Blender New System
	- Open a new 'General' project
	- Remove the cube object which is already selected by hitting the delete button (or right click on the object, select delete)
	- Add a UV Sphere by selecting Add => Mesh => UV Sphere, make sure you are in the Object Mode
	- While the UV Sphere is selected, add Empty Hair object on it by selecting Add => Curve => Empty Hair
	- Change the mode to Sculpt Mode	
	- Select 'Add' on the left-side menu bar
	- Make some changes to the hair curve settings to make it easier to work with (setting can be found at the top bar)
		- Change the Radius to 20 px to make the hair more dense
		- Change the Count to 100 to have enought hair strands to visually inspect them
		- In Curve Shape, change the Length to 1 m
	- Select the Comb tool and increase the Radius, for example to 80 px, for easier styling. 
	- Note that you can increase the hair resolution by increasing the Additional Subdivision by selecting Render Properties on the right-side menu
	
Exporting Hair Models as TFX
	- Select all the hair objects that are going to be exported. For multiple selections, hold the shift button and select hair objects by left 
	  clicking on them. Keep in mind that only the selected hair objects will be exported and other types of objects will be ignored even if selected.
	- Export the selected hair curve objects by going to File => Export => TressFX(.tfx) and hitting the 'Export TFX!' button. 
	  Important Notes: 
		- For each object a new save file window will be opened and objects will be exported as separate tfx files
		- In order to export bone influence weights, check the exportBones option
	  
Importing TFX Hair Models in O3DE
	- Open O3DE and add a new level if necessary
	- Make sure the following files are in the Assets folder of the AtomTressFX gem 
	  (the suggested path is {Engine Directory}/Gems/AtomTressFX/Assets/TestData/BlenderExporter):
		- sphere_actor.fbx, which is the fbx model of the UV Sphere we used in Blender to make sure the size of the mesh object we are using is 
		  the same in both environments. 
		  If this file is not already in the suggested folder, you might find it at {Engine Directory}/Gems/AtomTressFX/Tools/Blender, 
		  if not you can select, export and save the UV sphere as an fbx object from your Blender model.
		- Any tfx files exported from Blender
	- Add the sphere_actor object to the level by searching sphere_actor in the Asset Browser and dragging it to the Entity Outliner.
	- Select the sphere_actor entity in the Entity Outliner and and add the Atom Hair component by clicking Add Component in Entity Inspector 
	  (on the right side) and searching and selecting the Atom Hair 
	- Load the tfx hair model as the Hair Asset for the Atom Hair component
	- Done! The hair model is imported. You can repeat the same process to import more models.

Future Work
	Improvements
		- Rewriting the algorithm based on number of segments instead of number vertices. This will make the code more readible and the 
		  algorithm more robust.
		- Using a consitent naming convention for the code, either camelCase or dash-case
		- ExportTFX can be improved by modifying the ExportHelper class (parent class) invoke method to open the file explorer window from
   		  the execute method
		- Open a pop up window when a file is being overwritten. The current behavior is turning the file name text box red when saving the
 		  file name, which is Blender's behavior.
		- Add OK and Cancel buttons when warning the user about the changes that will be made to the hair strand vertices to give them the 
		  option to proceed or cancel, The current behavior is informing the user after performing the change. 
		- Add support for collision mesh
	 
	Known Issues
		- The unregister function does not work. The function bpy.ops.script.reload() is used as a temporary solution which slows down Blender
  		  after calling to the level that Blender needs to be restarted.