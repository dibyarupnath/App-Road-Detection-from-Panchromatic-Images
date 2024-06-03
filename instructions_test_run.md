--------------
**TESTING INSTRUCTIONS:**
**0.** From the root dir, after running the command: `python app.py` the app will start running locally.
- First the "Home Screen" will open, you click on the "Initialise" button to get to the "Detector".

**1.** Put the black and white image i sent you in the path: "static\test_data\36.jpg"

**2. **Put the test model I sent you on Whatsapp in the path: "models\\"
- Make sure you put it as a folder. so that finally it is: "models\Swin-T\ckpt.pth"

**3.** Now you are all set to run the app for testing. Test inputs must always be: 
- Input Satellite Image Path: static\\\\test_data\\\\36.jpg  with the double backslashes
- Road Detection Model: RoadSegNN (with Swin-T backbone)

**4.** Click "Detect Roads"