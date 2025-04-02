# Rummy AI bot

Here's a comprehensive guide for setting up and using the Rummy AI bot with your mobile device:

## Project Workflow

1. **Setup Phase**
   - Install required software and dependencies
   - Connect your Android device
   - Configure screen regions for your specific rummy game

2. **Training Phase**
   - Calibrate computer vision system
   - Run training sessions while the bot plays games
   - Monitor and adjust parameters as needed

3. **Deployment Phase**
   - Run the trained bot
   - Monitor performance
   - Make adjustments as needed

## Detailed Setup Instructions

### Step 1: Environment Setup

1. **Install Python Dependencies**:
   ```bash
   pip install numpy opencv-python pytesseract pillow matplotlib torch adbutils
   ```

2. **Install Android Debug Bridge (ADB)**:
   - Download Android SDK Platform Tools from Google's developer site
   - Add ADB to your system PATH

3. **Install Tesseract OCR**:
   - Download and install Tesseract OCR
   - Set the path in the code: `pytesseract.pytesseract.tesseract_cmd = '/path/to/tesseract'`

### Step 2: Mobile Device Connection

1. **Enable Developer Options on your Android device**:
   - Go to Settings > About Phone
   - Tap "Build Number" 7 times until you see "You are now a developer"
   - Go to Developer Options and enable "USB Debugging"

2. **Connect your device**:
   - Connect your device to your computer using a USB cable
   - When prompted on your device, allow USB debugging
   - Confirm connection by running `adb devices` in terminal/command prompt
   - You should see your device listed with a device ID

3. **Update the script with your device ID**:
   ```python
   DEVICE_ID = "your-device-id-here"  # Update this value
   ```

### Step 3: Game Calibration

1. **Capture Reference Screenshots**:
   ```python
   # Add this to the code to save reference screenshots
   def capture_reference_screenshots():
       controller = DeviceController()
       vision = RummyVision()
       screen = vision.capture_screen()
       if screen is not None:
           cv2.imwrite("reference_screen.png", screen)
           print("Reference screenshot saved!")
   
   # Run this function first
   capture_reference_screenshots()
   ```

2. **Calibrate Screen Regions**:
   - Open the reference screenshot in an image editor
   - Identify pixel coordinates for:
     - Cards in hand region
     - Draw pile location
     - Discard pile location
   - Update these regions in the code:
   ```python
   HAND_REGION = (x1, y1, x2, y2)  # Update with actual coordinates
   DRAW_PILE_REGION = (x1, y1, x2, y2)
   DISCARD_PILE_REGION = (x1, y1, x2, y2)
   ```

3. **Calibrate Card Templates**:
   - For optimal performance, create individual templates for each card in the game
   - Crop individual card images from your reference screenshots
   - Store these images in a folder structure, e.g., `templates/H/A.png` for Ace of Hearts
   - Update the `_load_card_templates` method with correct paths

### Step 4: Training the Bot

1. **Start with short training sessions**:
   ```python
   # Run initial training with fewer episodes
   player = RummyRLPlayer()
   player.train(num_episodes=10)
   ```

2. **Monitor the training progress**:
   - Check the generated `training_progress.png` file
   - Look for upward trends in episode rewards
   - If rewards are not improving, adjust reward functions or learning parameters

3. **Gradually increase training duration**:
   - Once initial results look promising, run longer training sessions
   - Save and backup model checkpoints periodically

### Step 5: Using the Trained Bot

1. **Run the bot in play mode**:
   ```python
   player = RummyRLPlayer()
   player.play(num_games=5)
   ```

2. **Observe and track performance**:
   - Monitor win rate
   - Check for any issues in card recognition
   - Ensure the bot is making sensible decisions

## Practical Usage Tips

1. **Use Virtual Devices for Initial Testing**:
   - Set up an Android emulator (like Android Studio's emulator)
   - Install your target rummy game
   - Test the system with the emulator before using real devices

2. **Automatic Game Navigation**:
   - Extend the DeviceController to handle game navigation:
   ```python
   def navigate_to_new_game(self):
       """Navigate through menus to start a new game"""
       # Sequence of taps to navigate menus
       self._tap((x1, y1))  # Tap "Play Again" button
       time.sleep(1)
       self._tap((x2, y2))  # Tap "Confirm" button
       time.sleep(3)  # Wait for game to load
   ```

3. **Card Recognition Improvements**:
   - Implement advanced image processing for better card recognition:
   ```python
   def enhance_card_visibility(self, card_image):
       """Preprocess card image for better recognition"""
       # Convert to grayscale
       gray = cv2.cvtColor(card_image, cv2.COLOR_BGR2GRAY)
       # Apply adaptive thresholding
       thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                     cv2.THRESH_BINARY, 11, 2)
       return thresh
   ```

4. **Error Handling**:
   - Add robust error handling to prevent crashes:
   ```python
   def safe_execute_action(self, action):
       """Safely execute an action with error handling"""
       try:
           self.controller.execute_action(action)
           return True
       except Exception as e:
           print(f"Error executing action: {e}")
           # Attempt recovery
           self._reset_game_state()
           return False
   ```

## Troubleshooting Common Issues

1. **ADB Connection Issues**:
   - Ensure USB debugging is enabled
   - Try different USB cables
   - Restart ADB server: `adb kill-server` followed by `adb start-server`

2. **Card Recognition Problems**:
   - Adjust lighting conditions (avoid glare)
   - Increase template matching threshold
   - Use multiple templates for each card

3. **Bot Making Poor Decisions**:
   - Adjust reward function to better reflect good rummy strategy
   - Increase training episodes
   - Add domain-specific knowledge to the decision logic

4. **Slow Performance**:
   - Optimize screenshot capture frequency
   - Use smaller image resolutions for processing
   - Consider using GPU acceleration for the neural network
