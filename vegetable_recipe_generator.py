import cv2
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from transformers import AutoImageProcessor, AutoModelForImageClassification
import time
import random

# Load model
processor = AutoImageProcessor.from_pretrained("jazzmacedo/fruits-and-vegetables-detector-36")
model = AutoModelForImageClassification.from_pretrained("jazzmacedo/fruits-and-vegetables-detector-36")

# Get the list of labels from the model's configuration
labels = list(model.config.id2label.values())

# Define the preprocessing transformation
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def detect_vegetables(frame, confidence_threshold=0.5):
    """Detect multiple vegetables in a frame using sliding windows"""
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w = frame.shape[:2]

    # Define grid for image segmentation
    # These values can be adjusted based on expected vegetable sizes
    grid_size = 5  # 5x5 grid
    detections = []
    detected_classes = set()
    colors = {}

    # For visualization - make a copy of the frame
    visualization_frame = frame.copy()

    # Process each grid cell
    for row in range(grid_size):
        for col in range(grid_size):
            # Calculate region boundaries
            x_start = int(col * w / grid_size)
            y_start = int(row * h / grid_size)
            x_end = int((col + 1) * w / grid_size)
            y_end = int((row + 1) * h / grid_size)

            # Extract region
            region = rgb_frame[y_start:y_end, x_start:x_end]

            # Convert to PIL image
            pil_region = Image.fromarray(region)

            # Preprocess and run inference
            input_tensor = preprocess(pil_region).unsqueeze(0)

            with torch.no_grad():
                outputs = model(input_tensor)

            # Get prediction and confidence
            probabilities = torch.nn.functional.softmax(outputs.logits, dim=1)
            confidence, predicted_idx = torch.max(probabilities, dim=1)
            confidence_value = confidence.item()
            predicted_label = labels[predicted_idx.item()]

            # Only keep detections above threshold
            if confidence_value > confidence_threshold:
                if predicted_label not in colors:
                    # Assign a random color for this class
                    colors[predicted_label] = (
                        np.random.randint(0, 255),
                        np.random.randint(0, 255),
                        np.random.randint(0, 255)
                    )

                # Store detection
                detections.append({
                    'label': predicted_label,
                    'confidence': confidence_value,
                    'region': (x_start, y_start, x_end, y_end)
                })

                # Keep track of unique classes
                detected_classes.add(predicted_label)

                # Draw rectangle around region
                cv2.rectangle(
                    visualization_frame,
                    (x_start, y_start),
                    (x_end, y_end),
                    colors[predicted_label],
                    2
                )

                # Add label
                label_text = f"{predicted_label}: {confidence_value * 100:.1f}%"
                cv2.putText(
                    visualization_frame,
                    label_text,
                    (x_start + 5, y_start + 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    colors[predicted_label],
                    2
                )

    # Return list of unique vegetables detected
    return list(detected_classes), visualization_frame, detections


def generate_recipe(ingredients):
    """Generate a simple recipe based on detected ingredients"""
    if not ingredients:
        return "No ingredients detected. Please show some vegetables to the camera."

    # Recipe templates with placeholders
    recipe_templates = [
        {
            "name": "{main_ingredient} Stir Fry",
            "description": "A quick and healthy stir fry featuring {ingredients_list}.",
            "instructions": [
                "Chop {ingredients_list} into bite-sized pieces.",
                "Heat oil in a wok or large pan over high heat.",
                "Add {main_ingredient} and stir-fry for 2 minutes.",
                "Add remaining ingredients and cook for 3-4 minutes until tender-crisp.",
                "Season with soy sauce, garlic, and ginger.",
                "Serve hot over rice or noodles."
            ]
        },
        {
            "name": "Roasted {main_ingredient} Medley",
            "description": "A delicious roasted vegetable dish with {ingredients_list}.",
            "instructions": [
                "Preheat oven to 425°F (220°C).",
                "Cut {ingredients_list} into chunks.",
                "Toss with olive oil, salt, pepper, and herbs.",
                "Spread on a baking sheet and roast for 25-30 minutes.",
                "Toss halfway through cooking time.",
                "Garnish with fresh herbs before serving."
            ]
        },
        {
            "name": "{main_ingredient} Soup",
            "description": "A comforting soup made with {ingredients_list}.",
            "instructions": [
                "Dice {ingredients_list} into small pieces.",
                "Sauté onions and garlic in a large pot until translucent.",
                "Add {main_ingredient} and other vegetables and cook for 5 minutes.",
                "Pour in vegetable broth and bring to a boil.",
                "Reduce heat and simmer for 20 minutes until vegetables are tender.",
                "Blend if desired, season to taste, and serve hot."
            ]
        },
        {
            "name": "Fresh {main_ingredient} Salad",
            "description": "A refreshing salad featuring {ingredients_list}.",
            "instructions": [
                "Wash and chop {ingredients_list} into bite-sized pieces.",
                "Combine all vegetables in a large bowl.",
                "Make a dressing with olive oil, lemon juice, salt, and pepper.",
                "Toss the vegetables with the dressing.",
                "Let sit for 10 minutes to allow flavors to meld.",
                "Serve chilled as a side dish or light meal."
            ]
        }
    ]

    # Select a random recipe template
    recipe_template = random.choice(recipe_templates)

    # Choose the main ingredient (the one with highest confidence or random)
    main_ingredient = random.choice(ingredients)

    # Format the ingredients list as a string
    if len(ingredients) > 1:
        ingredients_list = ", ".join(ingredients[:-1]) + " and " + ingredients[-1]
    else:
        ingredients_list = ingredients[0]

    # Fill in the recipe template
    recipe = {
        "name": recipe_template["name"].format(main_ingredient=main_ingredient),
        "description": recipe_template["description"].format(ingredients_list=ingredients_list),
        "ingredients": [f"{ingredient}" for ingredient in ingredients],
        "instructions": [
            instruction.format(
                main_ingredient=main_ingredient,
                ingredients_list=ingredients_list
            ) for instruction in recipe_template["instructions"]
        ]
    }

    # Format the recipe as a string
    recipe_text = f"RECIPE: {recipe['name']}\n\n"
    recipe_text += f"{recipe['description']}\n\n"
    recipe_text += "Ingredients:\n"
    for ingredient in recipe["ingredients"]:
        recipe_text += f"- {ingredient}\n"
    recipe_text += "\nInstructions:\n"
    for i, instruction in enumerate(recipe["instructions"]):
        recipe_text += f"{i + 1}. {instruction}\n"

    return recipe_text


# Initialize webcam
cap = cv2.VideoCapture(0)  # Use 0 for default webcam

# Check if the webcam is opened correctly
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Timing variables
start_time = time.time()
last_scan_time = 0
last_recipe_time = 0
warm_up_period = 5  # 5 seconds warm-up before starting to scan
scan_interval = 5  # Take one frame every 5 seconds
recipe_interval = 5  # Generate a new recipe every 5 seconds after detection
confidence_threshold = 0.5  # 50% confidence threshold

# Store the current recipe and detected vegetables
current_recipe = "Waiting for ingredients..."
detected_vegetables_list = []

print("Press 'q' to quit")
print(f"Warming up for {warm_up_period} seconds...")

# Create a separate window for the recipe
cv2.namedWindow("Recipe", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Recipe", 500, 600)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    if not ret:
        print("Error: Failed to capture image")
        break

    current_time = time.time()
    elapsed_time = current_time - start_time

    # Display countdown during warm-up period
    if elapsed_time < warm_up_period:
        countdown = f"Starting in: {max(1, int(warm_up_period - elapsed_time))}s"
        cv2.putText(frame, countdown, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow('Multi-Vegetable Detection', frame)

        # Create blank recipe window during warm-up
        recipe_image = np.ones((600, 500, 3), dtype=np.uint8) * 255
        cv2.putText(recipe_image, "Waiting for ingredients...", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        cv2.imshow('Recipe', recipe_image)

        # Check for quit command
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue

    # After warm-up, scan at specified intervals
    if current_time - last_scan_time >= scan_interval:
        last_scan_time = current_time

        # Detect vegetables in the frame
        detected_vegetables, annotated_frame, vegetable_details = detect_vegetables(
            frame,
            confidence_threshold=confidence_threshold
        )

        # Update the global list
        if detected_vegetables:
            detected_vegetables_list = detected_vegetables

        # Print list of detected vegetables
        if detected_vegetables:
            print(f"Detected vegetables: {', '.join(detected_vegetables)}")
            for veg in vegetable_details:
                print(f"  - {veg['label']}: {veg['confidence'] * 100:.1f}%")
        else:
            print("No vegetables detected with sufficient confidence")

        # Use the annotated frame for display
        frame = annotated_frame

        # Show detected vegetables list on frame
        if detected_vegetables:
            label_text = f"Detected: {', '.join(detected_vegetables)}"
            cv2.putText(frame, label_text, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "No vegetables detected", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Generate recipe periodically if we have ingredients
    if (current_time - last_recipe_time >= recipe_interval) and detected_vegetables_list:
        last_recipe_time = current_time
        current_recipe = generate_recipe(detected_vegetables_list)
        print("\nNEW RECIPE GENERATED:")
        print(current_recipe)

    # Display next scan countdown
    time_to_next_scan = max(0, scan_interval - (current_time - last_scan_time))
    next_scan_text = f"Next scan in: {time_to_next_scan:.1f}s"
    cv2.putText(frame, next_scan_text, (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 200), 2)

    # Create recipe window
    recipe_image = np.ones((600, 500, 3), dtype=np.uint8) * 255  # White background

    # Split recipe text into lines and add to image
    y_position = 40
    line_height = 30
    for line in current_recipe.split('\n'):
        # Display recipe title in larger font
        if line.startswith("RECIPE:"):
            cv2.putText(recipe_image, line, (20, y_position),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
            y_position += int(line_height * 1.5)
        else:
            cv2.putText(recipe_image, line, (20, y_position),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
            y_position += line_height

    # Display windows
    cv2.imshow('Multi-Vegetable Detection', frame)
    cv2.imshow('Recipe', recipe_image)

    # Check for quit command
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release webcam and close windows
cap.release()
cv2.destroyAllWindows()