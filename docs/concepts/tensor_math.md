# Tensor Math & Optimization Simplified

This compiler deals with "Tensors." If you aren't a data scientist, here is what that means.

## What is a Tensor?
A **Tensor** is just a grid of numbers.
*   **0D Tensor:** A single number (Scalar). `5`
*   **1D Tensor:** A list of numbers (Vector). `[1, 2, 3]`
*   **2D Tensor:** A spreadsheet or a grayscale image (Matrix).
*   **3D Tensor:** A stack of images (Video) or a color image (RGB).

## The Heavy Lifter: 2D Convolution
One of the most important operations in AI (especially for computer vision) is **2D Convolution**.

### The Analogy: The Sliding Window
Imagine you are looking at a picture through a small square hole cut in a piece of paper (a 3x3 window).
1.  You place the window over the top-left corner of the image.
2.  You multiply the pixels you see by some specific weights (filters).
3.  You sum them up to get a single number.
4.  You slide the window one step to the right and repeat.

This detects features like edges, corners, or cats.

### Why is it slow?
Ideally, this is fast. But computers have complex memory hierarchies (Cache).
If you read pixels essentially at random (jumping between rows), the computer spends more time waiting for memory than doing math.

A naive convolution implementation might look like this ($O(N^4)$):
```python
for batch in images:
  for row in height:
    for col in width:
      for channel in colors:
         # ... calculate ...
```
This involves billions of operations for a single image.

## Optimization: Tiling (Blocking)
To make this faster, we use **Tiling**.

Imagine trying to read a book, but every time you finish a line, you have to walk to the library to get the next line. That's naive code.

**Tiling** means you grab a whole chapter (a "Tile" or "Block" of data), bring it to your desk (CPU Cache), and finish reading it before going back to the library.

NeuroJIT asks the AI to figure out the best "Chapter Size" (Tile Size) for your specific CPU automatically.
