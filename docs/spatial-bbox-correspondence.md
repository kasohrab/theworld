# Spatial Bounding Box Correspondence in OpenSpatialDataset

**Author:** Generated from dataset analysis
**Date:** 2025-01-08
**Purpose:** Document how bounding boxes correspond to region references in multi-turn spatial conversations

## TL;DR

**The bbox array is pre-ordered to match sequential `<mask>` token appearance in the conversation.**

- Nth `<mask>` token → bbox[N] in the array
- Total bbox entries = Total `<mask>` count
- This is a **structural invariant** of the dataset format

## SpatialRGPT vs TheWorld Approach

### SpatialRGPT (Original Paper)

**Architecture:**
- Uses **mask tensors** (spatial feature maps) for each region
- Region extractor (MaskPooling) extracts features from masked areas
- Replaces `<mask>` token embeddings with region features

**Region Labeling:**
- Uses **turn-local** numbering (resets each turn)
- Text says "Region [0], Region [1]" but these are turn-specific
- Mask tensors provide the actual grounding (model knows which physical region)

**Example:**
```
Turn 3 input:
  Text: "Does Region [0] have lesser width than Region [1]?"
  Masks: [mask_for_bbox[3], mask_for_bbox[4]]  ← Explicit grounding!

Model processing:
  - "Region [0]" → replaced with features from mask_for_bbox[3]
  - "Region [1]" → replaced with features from mask_for_bbox[4]
  - Turn-local labels are just placeholders
```

**Pros:**
- Precise spatial grounding via mask tensors
- Turn-local numbering simplifies annotation

**Cons:**
- Requires mask generation pipeline
- Complex region extractor architecture
- Cannot use standard vision encoders

---

### TheWorld (Our Approach)

**Architecture:**
- Uses **drawn bounding boxes** on image (visual labels)
- Standard vision encoder (SigLIP) processes labeled image
- No mask tensors or region extractors needed

**Region Labeling:**
- Uses **global** numbering (consistent across all turns)
- All unique regions labeled [0, 1, 2, ...] on the image
- Text references must match visual labels

**Example:**
```
Image: Shows 5 bboxes labeled "Region [0]", "Region [1]", ..., "Region [4]"

Turn 3 input:
  Text: "Does Region [1] have lesser width than Region [2]?"
  Visual: Model sees the drawn labels on the image

Model processing:
  - "Region [1]" → refers to bbox labeled "Region [1]" in image
  - "Region [2]" → refers to bbox labeled "Region [2]" in image
  - Global labels ensure consistency
```

**Pros:**
- Simple: just draw bboxes on images
- Works with standard vision encoders
- No additional training infrastructure
- Easier to debug (visual inspection)

**Cons:**
- Requires global ID conversion (turn-local → global)
- Label positioning complexity for overlapping bboxes
- Relies on OCR-like ability to read bbox labels

---

### Key Differences

| Aspect | SpatialRGPT | TheWorld |
|--------|-------------|----------|
| **Region Grounding** | Mask tensors | Drawn bbox labels |
| **Region Numbering** | Turn-local [0, 1, ...] | Global [0, 1, 2, ...] |
| **Vision Encoder** | Custom region extractor | Standard (SigLIP) |
| **Dataset Format** | Original (turn-local) | Converted (global) |
| **Label Overlap Handling** | N/A (no visual labels) | Smart positioning algorithm |
| **Complexity** | High (mask pipeline) | Low (draw + encode) |

---

### Why TheWorld Needs Global IDs

**Problem with turn-local IDs:**
```
Image shows: Region [0], Region [1], Region [2], Region [3], Region [4]
Turn 3 asks: "Does Region [0] have lesser width than Region [1]?"

Model confusion:
- Which "Region [0]"? The one labeled [0] on image, or turn-local meaning [1]?
- Visual label says [0], text says [0], but they mean different regions!
```

**Solution with global IDs:**
```
Image shows: Region [0], Region [1], Region [2], Region [3], Region [4]
Turn 3 asks: "Does Region [1] have lesser width than Region [2]?"

Model clarity:
- Text "Region [1]" → refers to bbox labeled [1] on image ✓
- Text "Region [2]" → refers to bbox labeled [2] on image ✓
- Perfect alignment between visual and textual references
```

## Dataset Structure

### Data Format

```json
{
  "id": 0,
  "filename": "image_001",
  "conversations": [
    {"from": "human", "value": "Does <mask> <depth> have greater width than <mask> <depth>?"},
    {"from": "gpt", "value": "In fact, Region [0] might be narrower than Region [1]."},
    {"from": "human", "value": "Which is taller, <mask> <depth> or <mask> <depth>?"},
    {"from": "gpt", "value": "Standing taller is Region [0]."}
  ],
  "bbox": [
    [100, 150, 300, 400],  // bbox[0] → 1st <mask> in conversation
    [400, 200, 600, 500],  // bbox[1] → 2nd <mask> in conversation
    [100, 150, 300, 400],  // bbox[2] → 3rd <mask> in conversation (duplicate!)
    [400, 200, 600, 500]   // bbox[3] → 4th <mask> in conversation (duplicate!)
  ]
}
```

### Key Properties

1. **Positional Correspondence**:
   ```
   <mask> token index (0-based, sequential through entire conversation)
   → bbox array index
   ```

2. **Invariant (VERIFIED)**:
   ```python
   len(bbox_array) == sum(conv['value'].count('<mask>') for conv in conversations if conv['from'] == 'human')
   ```

3. **Bbox Duplication**:
   - Same physical region appears multiple times in bbox array
   - One entry per `<mask>` appearance in conversation
   - Example: Region A mentioned in turns 1, 3, 5 → appears 3 times in bbox array

## Verification

Tested on 5 samples from OpenSpatialDataset:

| Sample | Total `<mask>` | Total `bbox` | Match |
|--------|----------------|--------------|-------|
| 001    | 19             | 19           | ✓     |
| 002    | 18             | 18           | ✓     |
| 003    | 18             | 18           | ✓     |
| 004    | 18             | 18           | ✓     |
| 005    | 20             | 20           | ✓     |

**Invariant holds 100% across all tested samples.**

## Algorithm: Determine Which Bboxes a Turn Uses

```python
def get_bbox_indices_for_turn(conversations, turn_index):
    """
    Get which bbox indices a specific turn uses.

    Args:
        conversations: List of conversation turns
        turn_index: 0-based turn index (0, 2, 4... for user turns)

    Returns:
        List of bbox indices used by this turn
    """
    # Count masks in all previous turns
    mask_counter = 0
    for i in range(0, turn_index, 2):  # Only user turns
        mask_counter += conversations[i]['value'].count('<mask>')

    # Count masks in current turn
    mask_count_this_turn = conversations[turn_index]['value'].count('<mask>')

    # This turn uses bbox[mask_counter : mask_counter + mask_count_this_turn]
    return list(range(mask_counter, mask_counter + mask_count_this_turn))
```

### Example Execution (Sample 002)

```python
conversations = [  # 10 turns total
    {"from": "human", "value": "... <mask> <depth> ... <mask> <depth> ..."},  # Turn 0: 2 masks
    {"from": "gpt", "value": "..."},
    {"from": "human", "value": "... <mask> <depth> ..."},                      # Turn 2: 1 mask
    {"from": "gpt", "value": "..."},
    {"from": "human", "value": "... <mask> <depth> ... <mask> <depth> ..."},  # Turn 4: 2 masks
    # ... more turns
]

# Turn 0 (index 0): uses bbox[0:2] → [0, 1]
# Turn 1 (index 2): uses bbox[2:3] → [2]
# Turn 2 (index 4): uses bbox[3:5] → [3, 4]  ← This is how we know!
```

## Unique Regions vs Bbox Array

### Bbox Array Contains Duplicates

Sample 002 has:
- **18 bbox entries** (one per `<mask>` appearance)
- **5 unique regions** (deduplicated by coordinate equality)

```python
# Deduplicate to find unique physical regions
unique_bboxes = []
bbox_to_region_id = {}

for idx, bbox in enumerate(bbox_array):
    bbox_tuple = tuple(bbox)

    # Check if this bbox already exists
    if bbox_tuple in [tuple(u) for u in unique_bboxes]:
        # Find which region this duplicates
        region_id = [tuple(u) for u in unique_bboxes].index(bbox_tuple)
    else:
        # New unique region
        region_id = len(unique_bboxes)
        unique_bboxes.append(bbox)

    bbox_to_region_id[idx] = region_id

# Result for sample 002:
# bbox[0,1,2] → Region 0 (same physical box)
# bbox[3,5,7,10,14] → Region 1 (same physical box)
# bbox[4,9,11,15] → Region 2
# bbox[6,8,12,16] → Region 3
# bbox[13,17] → Region 4
```

## Region ID Mapping: Turn-Local vs Global

### Original Dataset (SpatialRGPT with Mask Tensors)

SpatialRGPT uses **turn-local** region numbering:
- Each turn resets to Region [0], [1], [2]...
- Mask tensors provide the actual spatial grounding
- Text "Region [0]" is turn-specific, resolved via mask tensor

**Example (Turn 3):**
```
Input:
  - Text: "Does Region [0] have lesser width than Region [1]?"
  - Masks: [mask_for_bbox[3], mask_for_bbox[4]]  ← Explicit correspondence!

Model knows: "Region [0]" in this turn = mask_for_bbox[3]
```

### TheWorld (Drawn Bboxes, No Mask Tensors)

TheWorld uses **global** region numbering:
- ALL unique regions labeled [0-4] on the image
- Text must reference these global labels
- Visual bbox labels provide spatial grounding

**Example (Turn 3, CORRECTED):**
```
Input:
  - Image: Has boxes labeled "Region [0]", "Region [1]", "Region [2]", "Region [3]", "Region [4]"
  - Text: "Does Region [1] have lesser width than Region [2]?"  ← Must match image labels!

Model sees: Text references the actual drawn boxes
```

## Fix Required for TheWorld

**Current (WRONG):**
```python
def replace_region_tokens(text):
    """Sequential replacement - resets per turn."""
    region_idx = 0
    while "<mask> <depth>" in text:
        text = text.replace("<mask> <depth>", f"Region [{region_idx}]", 1)
        region_idx += 1  # 0, 1, 2... per turn
    return text

# Turn 3 result: "Region [0]" and "Region [1]" (turn-local)
# But image shows: Region [1] and Region [2] (global)
# MISMATCH!
```

**Fixed (CORRECT):**
```python
def replace_region_tokens_with_global_ids(conversations, bbox_array):
    """Replace <mask> tokens with global region IDs."""
    # 1. Deduplicate bbox array to get unique regions
    unique_bboxes = []
    bbox_to_global_region = {}
    for idx, bbox in enumerate(bbox_array):
        bbox_tuple = tuple(bbox)
        if bbox_tuple not in [tuple(u) for u in unique_bboxes]:
            region_id = len(unique_bboxes)
            unique_bboxes.append(bbox)
        else:
            region_id = [tuple(u) for u in unique_bboxes].index(bbox_tuple)
        bbox_to_global_region[idx] = region_id

    # 2. Replace <mask> tokens sequentially with global IDs
    mask_counter = 0
    for i in range(0, len(conversations), 2):  # User turns only
        text = conversations[i]['value']
        mask_count = text.count('<mask>')

        for j in range(mask_count):
            global_region_id = bbox_to_global_region[mask_counter]
            text = text.replace('<mask> <depth>', f'Region [{global_region_id}]', 1)
            mask_counter += 1

        conversations[i]['value'] = text

    return conversations

# Turn 3 result: "Region [1]" and "Region [2]" (global)
# Image shows: Region [1] and Region [2] (global)
# MATCH!
```

## Complete Transformation Pipeline

TheWorld applies the following transformations to SpatialRGPT dataset for training:

### Step 1: Bbox Deduplication and Global Mapping

```python
# Input: bbox array with duplicates (one per <mask> appearance)
bbox_array = [bbox0, bbox1, bbox0, bbox1, ...]  # 18 entries

# Deduplicate to find unique physical regions
unique_bboxes = []
bbox_to_global_region = {}

for idx, bbox in enumerate(bbox_array):
    bbox_tuple = tuple(bbox)
    if bbox_tuple not in [tuple(u) for u in unique_bboxes]:
        region_id = len(unique_bboxes)  # Assign next global ID
        unique_bboxes.append(bbox)
    else:
        region_id = [tuple(u) for u in unique_bboxes].index(bbox_tuple)
    bbox_to_global_region[idx] = region_id

# Result: bbox_to_global_region = {0:0, 1:1, 2:2, 3:1, 4:2, ...}
#         Maps bbox array index → global region ID
```

### Step 2: Draw Unique Bboxes with Global Labels

```python
# Draw ONLY unique regions on image (avoid clutter)
for i, bbox in enumerate(unique_bboxes):
    draw_bbox(bbox, label=f"Region [{i}]")

# Image now shows: Region [0], Region [1], Region [2], ...
```

### Step 3: Convert User Questions (Replace `<mask>` tokens)

```python
# Original: "Does <mask> <depth> have lesser width than <mask> <depth>?"
# Turn 3 uses mask[3] and mask[4]

global_mask_counter = 0
turn_local_to_global = {}  # Will store {0: 1, 1: 2} for this turn

for user_turn in conversations:
    turn_local_counter = 0

    while "<mask> <depth>" in user_turn['value']:
        # Get global region ID for this mask
        global_region_id = bbox_to_global_region[global_mask_counter]

        # Track turn-local→global mapping (for assistant response)
        turn_local_to_global[turn_local_counter] = global_region_id

        # Replace with global ID
        user_turn['value'] = user_turn['value'].replace(
            "<mask> <depth>",
            f"Region [{global_region_id}]",
            1
        )

        global_mask_counter += 1
        turn_local_counter += 1

# Result: "Does Region [1] have lesser width than Region [2]?"
#         turn_local_to_global = {0: 1, 1: 2}
```

### Step 4: Convert Assistant Answers (Replace turn-local IDs)

```python
# Original answer has turn-local IDs: "In fact, Region [0] might be wider than Region [1]."
# We need to convert to global: "In fact, Region [1] might be wider than Region [2]."

for assistant_turn in conversations:
    def replace_region_id(match):
        turn_local_id = int(match.group(1))  # Extract "0" or "1"
        if turn_local_id in turn_local_to_global:
            global_id = turn_local_to_global[turn_local_id]  # 0→1, 1→2
            return f"Region [{global_id}]"
        return match.group(0)

    # Replace all "Region [N]" with global IDs
    assistant_turn['value'] = re.sub(
        r'Region \[(\d+)\]',
        replace_region_id,
        assistant_turn['value']
    )

# Result: "In fact, Region [1] might be wider than Region [2]."
```

### Complete Example (Sample 002, Turn 3)

**Original Data:**
```python
{
  "bbox": [bbox0, bbox1, bbox2, bbox3, bbox4, ...],  # 18 total
  "conversations": [
    # Turn 1, Turn 2...
    {"from": "human", "value": "Does <mask> <depth> have lesser width than <mask> <depth>?"},
    {"from": "gpt", "value": "In fact, Region [0] might be wider than Region [1]."},
    # More turns...
  ]
}
```

**After Transformation:**
```python
{
  "unique_bboxes": [bbox0, bbox1, bbox2, bbox3, bbox4],  # 5 unique
  "messages": [
    # Turn 1, Turn 2...
    {"role": "user", "content": "Does Region [1] have lesser width than Region [2]?"},
    {"role": "assistant", "content": "In fact, Region [1] might be wider than Region [2]."},
    # More turns...
  ]
}
```

**Visual Rendering:**
- Image shows 5 bboxes labeled: Region [0], Region [1], Region [2], Region [3], Region [4]
- Turn 3 question asks about Region [1] and Region [2] (matches image!)
- Turn 3 answer refers to Region [1] and Region [2] (consistent!)

## Summary

**For TheWorld to work correctly with drawn bboxes:**

1. ✅ Draw ALL unique regions on image with global labels [0, 1, 2, ...]
2. ✅ Use positional correspondence to map Nth `<mask>` → bbox[N]
3. ✅ Deduplicate bbox array to find unique regions
4. ✅ Replace `<mask>` tokens in questions with **global region IDs**
5. ✅ Replace turn-local IDs in answers with **global region IDs**
6. ✅ Ensure text references match the drawn bbox labels

This ensures the model sees consistent region references between:
- **Visual**: Drawn bboxes with global labels
- **Questions**: Region references using global IDs
- **Answers**: Region references using global IDs

**Implementation**: See `python/theworld/datasets/spatial_rgpt.py` lines 368-544

## Smart Label Positioning for Overlapping Bboxes

When bboxes overlap or are close together, naive label positioning (always top-left) causes labels to obscure each other.

### Problem

**Example (Sample 002):**
- Region 0: [373, 357, 456, 435]
- Region 2: [365, 354, 463, 435]

These bboxes are very close → labels at default top-left position overlap and become unreadable.

### Solution

TheWorld implements smart label positioning with multiple fallback strategies:

1. **Pre-calculate all label positions** before drawing (not one-by-one)
2. **Try multiple position strategies** for each label (in priority order):
   - `top-outside`: Above bbox (default, preferred)
   - `bottom-outside`: Below bbox
   - `top-inside`: Inside bbox at top
   - `right-outside`: To the right of bbox
   - `left-outside`: To the left of bbox
   - `bottom-inside`: Inside bbox at bottom
3. **Check for overlaps** with previously-placed labels
4. **Use first non-overlapping position** found
5. **Fallback**: If all strategies fail, stagger by small offset

### Implementation

```python
def _find_non_overlapping_positions(bboxes, labels, image_size, font):
    """Find non-overlapping label positions for all bboxes."""
    label_positions_bboxes = []  # Track placed label areas
    label_positions_xy = []      # Final (x, y) positions

    strategies = ["top-outside", "bottom-outside", "top-inside", "right-outside", ...]

    for bbox, label in zip(bboxes, labels):
        # Try each strategy
        for strategy in strategies:
            pos = calculate_position(bbox, label_size, strategy)
            label_bbox = (pos[0], pos[1], pos[0] + label_width, pos[1] + label_height)

            # Check overlap with existing labels
            overlaps = any(labels_overlap(label_bbox, existing) for existing in label_positions_bboxes)

            if not overlaps:
                # Found non-overlapping position!
                label_positions_xy.append(pos)
                label_positions_bboxes.append(label_bbox)
                break

        # Fallback: stagger by offset if all strategies failed
        if not found:
            pos = default_position + (i * 8, i * 8)  # Diagonal offset
            label_positions_xy.append(pos)

    return label_positions_xy
```

**Key Features:**
- **Greedy algorithm**: Places labels in order, trying strategies until non-overlapping
- **Multiple fallbacks**: 6 position strategies before resorting to offset
- **Image boundary aware**: Clamps positions to keep labels visible
- **Works for any number of overlapping bboxes**

**Result:**
- Sample 002 with 5 overlapping regions → All labels clearly visible
- Labels positioned strategically (some above, some below, some inside)
- No manual intervention needed

**Implementation**: See `python/theworld/datasets/bbox_utils.py` lines 8-126
