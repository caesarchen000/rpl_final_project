# ContactGen Poster Content Outline

## Layout: 3-4 Columns, Portrait or Landscape

---

## SECTION 1: TITLE & AUTHORS (Top, Full Width)

**Title (Large, Bold)**:
```
ContactGen: Generative Contact Modeling for Grasp Generation
```

**Authors** (if applicable):
```
Your Name, Collaborators
Institution Name
```

**Contact/QR Code** (optional):
- GitHub: github.com/stevenlsw/contactgen
- Paper: arxiv.org/abs/2310.03740

---

## SECTION 2: INTRODUCTION / MOTIVATION (Left Column, Top)

**Heading**: "Problem & Motivation"

**Content** (2-3 bullet points):
- Generate realistic human hand grasps for 3D objects
- Object-centric contact representation enables diverse, high-fidelity grasps
- Applications: robotics, AR/VR, human-computer interaction

**Visual**: 
- Image of diverse grasp examples (if available)
- Or: "Input: 3D Object → Output: Hand Grasp"

---

## SECTION 3: METHOD / ARCHITECTURE (Center-Left, Large)

**Heading**: "ContactGen Architecture"

**Content** (Concise):
- **Dual-Branch VAE**: Contact encoder/decoder + Partition encoder/decoder
- **Input**: Object point cloud → PointNet++ features
- **Output**: Contact map (probability) + Partition map (16 hand parts)
- **Latent Space**: Regularized to N(0,1) for diverse sampling

**Visual** (Most Important):
- Architecture diagram showing:
  - Input: Object point cloud
  - Encoders: Contact Encoder, Part Encoder
  - Latent: z_contact, z_part (with N(0,1) regularization)
  - Decoders: Contact Decoder, Part Decoder
  - Output: Contact Map, Partition Map

**Key Formula** (if space):
```
Contact Map = sigmoid(ContactDecoder(Feat, z_contact))
Partition Map = softmax(PartDecoder(Feat, z_part, z_contact))
```

---

## SECTION 4: HAND PART STRUCTURE (Center-Right, Small)

**Heading**: "16 Hand Parts"

**Content**:
- Palm (1) + 5 Fingers × 3 segments = 16 parts
- Each finger: base → middle → tip

**Visual**:
- Table or diagram showing:
  - Part ID 0: Palm
  - Part IDs 1-3: Index finger
  - Part IDs 4-6: Middle finger
  - Part IDs 7-9: Ring finger
  - Part IDs 10-12: Pinky
  - Part IDs 13-15: Thumb

**Color Legend** (small):
- Show color swatches for each part (brown, yellow, green, blue, purple, red)

---

## SECTION 5: WORKFLOW / PIPELINE (Center, Medium)

**Heading**: "Pipeline"

**Content** (Step-by-step):
1. **Input**: 3D object mesh (.obj/.ply)
2. **Generate Maps**: ContactGen inference → contact_map.npy, part_hard.npy
3. **Visualize**: Heatmap, partition map, combined visualization
4. **Optimize Grasp**: Hand pose optimization → grasp_0.obj

**Visual**:
- Flow diagram:
  ```
  Object Mesh
      ↓
  ContactGen Model
      ↓
  Contact Map + Partition Map
      ↓
  Visualization + Grasp Optimization
      ↓
  Colored Meshes + Hand Grasp
  ```

**Code Example** (small font):
```bash
python generate_all_visualizations.py \
  --obj_path object.obj \
  --output_dir results/
```

---

## SECTION 6: OUTPUT FILES & DATA STRUCTURES (Right Column, Top)

**Heading**: "Output Data Formats"

**Content** (Table format):

| File | Shape | Description |
|------|-------|-------------|
| `contact_map.npy` | [N] | Contact probabilities (0-1) |
| `part_hard.npy` | [N] | Hand part IDs (0-15) |
| `part_logits.npy` | [N, 16] | Raw logits for 16 parts |
| `final.npy` | [N, 16] | One-hot contact probabilities |
| `partition_contact.npy` | [N, 3] | RGB colors (part × contact) |

**Size Reference** (small):
- For N=2048 points: ~0.5 MB total

**Visualization Files**:
- `heatmap.obj` - Contact probability
- `partition.obj` - Hand part assignments
- `partition_contact.obj` - Combined visualization
- `grasp_0.obj` - Optimized hand mesh

---

## SECTION 7: RESULTS / VISUALIZATIONS (Center-Bottom, Large)

**Heading**: "Results"

**Content** (Visual-heavy section):

**Show 3-4 example objects with**:
1. **Input**: Original object mesh
2. **Contact Heatmap**: Red = high contact, blue = low
3. **Partition Map**: 16 colors showing hand part assignments
4. **Combined**: Partition colors × contact brightness
5. **Grasp**: Hand mesh in optimized pose

**Layout Options**:
- **Option A**: 3-4 objects in a row, each showing all 5 visualizations
- **Option B**: 1-2 objects with larger images showing the full pipeline

**Caption** (small):
- "ContactGen generates diverse, realistic grasps for various object shapes"

---

## SECTION 8: KEY CONTRIBUTIONS (Right Column, Middle)

**Heading**: "Key Features"

**Content** (Bullet points):
- ✓ **Object-centric representation**: Contact maps defined on object surface
- ✓ **16-part hand model**: Detailed finger segment assignments
- ✓ **Diverse generation**: VAE latent space enables multiple grasp samples
- ✓ **End-to-end pipeline**: From object mesh to colored visualizations
- ✓ **Multiple output formats**: .npy data files + .obj visualizations

---

## SECTION 9: TECHNICAL DETAILS (Right Column, Bottom, Small)

**Heading**: "Technical Specifications"

**Content** (Compact):
- **Model**: Dual-branch VAE with PointNet++ backbone
- **Input**: Object point cloud (N points, default N=2048)
- **Output**: Contact map [N], Partition map [N, 16]
- **Hand Parts**: 16 parts (palm + 5 fingers × 3 segments)
- **Optimization**: Gradient-based pose optimization with contact/penetration/UV losses

**File Sizes** (very small):
- Total data: ~0.5 MB per object (N=2048)
- Visualization: ~100-500 KB per .obj file

---

## SECTION 10: CONCLUSION / FUTURE WORK (Bottom, Optional)

**Heading**: "Conclusion & Future Work"

**Content** (2-3 sentences):
- ContactGen provides a complete framework for grasp generation with rich contact and partition representations
- Future: Real-time inference, multi-hand grasps, dynamic grasp sequences

**References** (very small):
- Liu et al., "ContactGen: Generative Contact Modeling for Grasp Generation", ICCV 2023
- Code: github.com/stevenlsw/contactgen

---

## DESIGN TIPS:

### Visual Hierarchy:
1. **Largest**: Architecture diagram, Results visualizations
2. **Medium**: Workflow diagram, Hand part structure
3. **Small**: Tables, code examples, technical details

### Color Scheme:
- Use the hand part colors (brown, yellow, green, blue, purple, red) as accent colors
- Keep background light/white for readability
- Use high contrast for text

### Text Guidelines:
- **Headings**: 36-48pt font
- **Body text**: 18-24pt font (readable from 3-4 feet away)
- **Captions**: 14-16pt font
- **Code**: 12-14pt monospace font

### Layout Suggestions:
- **Portrait (A0/A1)**: 3-4 columns, top to bottom flow
- **Landscape (A0/A1)**: 4-5 columns, left to right flow
- Leave white space (don't overcrowd)
- Use borders/boxes to separate sections

### Must-Have Visuals:
1. ✅ Architecture diagram (most important)
2. ✅ Results showing input → output pipeline
3. ✅ Hand part color legend
4. ✅ Workflow diagram

### Optional but Helpful:
- Comparison with other methods
- Quantitative results/metrics
- Ablation study
- Failure cases (if space)

---

## QUICK POSTER CHECKLIST:

- [ ] Title and authors clearly visible
- [ ] Architecture diagram included
- [ ] At least 2-3 result visualizations
- [ ] Hand part structure/legend shown
- [ ] Workflow/pipeline diagram
- [ ] Data format table
- [ ] Key contributions listed
- [ ] References included
- [ ] Text readable from 3-4 feet away
- [ ] Good use of white space
- [ ] Consistent color scheme



