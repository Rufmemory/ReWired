# ReWired
There you go dear lovely git folks! 

This is a late tribute to the pygame module envisioned by Petercollingridge 
 I named it ReWired as its corresponding Main class was elegantly named Wireframe. 

This module is now capable of visualising self made three dimensional shapes,
 by using the py Pygame module, declaring and groupping nodes, 
 generating vectors and matrices, and finally applying unit vectors and 
 common matrix operations we get to display predefined 3d objects, 
 along with customising the scaling, rotation, and lightning factor.

I have myself invested quite some time in setting up such a functional 3d-Engine, 
 with the allready existing pygame, Vec3D methods and various documentations. 

After looking a bit further on github i stumbled upon his old yet wonderfull Repo, 
 and by applying my own code drafts, calculus and linear algebra knowledge, 
 it quickly clarified a lot of the "hard" questions ive previously had around 
 setting up such an environment, preferably using PyGame.

As i cannot forget his substantial addition to my understanding, 
 and to the allready existing and wonderfull github community,
 I have thereby created this repo, as a tribute.
 It should hopefully fix a great deal of issues, 
 and as i did have to change and correct a lot of errors.

YET! There are still some things that i have changed in my newer version "GridMod"
 and should preferably be changed if forking this or his (Pygame-3D) repo for best practice purposes...

Such as:
- variable_names/functions(): they should prefferably allways be lowercased in order to distinguish them from Classes
                               or other Important methods and Handles from dependencies and imported packages 
                              such as the sweet pygame were using (ex: Key_Attributes = K_up)
- Renaming most function names, making their meaning more familliar (ex: translationMatrix() = translate_identity())
- Incorporating all var declarations and functions into their corresponding Classes
- Transforming more lists into tupples...
- Changing Key Binding

Feel free to let me know if you want to add or ask anything,
 perhaps you could have a look at my newest related project called GridMod! 
 the GridMod Project is all initialized by classes, and other personal additions/changes.
 And it is be the main 3d-engine concept im working on at the moment, 
 trying to implement kinetics and motion...

Suggessting what could further be accomplished, to make this happen would be awesome, 
Since this really took some great ammount of time as well to convert anf fix properly,
you are definetly also welcome to simply tell me if you liked it!!
