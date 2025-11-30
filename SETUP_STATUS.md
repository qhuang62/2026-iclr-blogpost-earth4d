# Earth4D ICLR Blog Post - Setup Status

## ✅ What's Been Done

### 1. Repository Setup
- ✅ Forked ICLR repository to: https://github.com/qhuang62/2026-iclr-blogpost-earth4d
- ✅ Cloned to: `/scratch/qhuang62/2026-iclr-blogpost-earth4d/`

### 2. Blog Post Files
- ✅ Blog post copied to: `_posts/2026-04-27-earth4d-world-models.md` (31KB, 4500 words)
- ✅ Bibliography copied to: `assets/bibliography/2026-04-27-earth4d-world-models.bib` (11 references)
- ✅ Placeholder images created in: `assets/img/2026-04-27-earth4d-world-models/`

### 3. Placeholder Images Created (for testing)
All 5 images are now in place with **placeholder content**:
- `earth4d_architecture.png` (24KB) - Blue box saying "Earth4D Architecture Placeholder"
- `lfmc_results.png` (21KB) - 3-panel placeholder
- `rgb_reconstruction.png` (9.9KB) - 4-panel placeholder
- `collision_heatmap.png` (33KB) - Heatmap with random data
- `compression_tradeoff.png` (33KB) - Scatter plot with actual data points

---

## 🎯 Next Steps

### Step 1: Test the Blog Locally (RECOMMENDED)

**Option A: Using Docker (Easiest)**
```bash
cd /scratch/qhuang62/2026-iclr-blogpost-earth4d

# Run with docker-compose
docker-compose up

# Or if that doesn't work:
docker run --rm -it -p 8080:8080 -v $(pwd):/srv/jekyll amirpourmand/al-folio:v0.14.6

# Open in browser: http://localhost:8080/2026/blog/2026/earth4d-world-models/
```

**Option B: Using Bundle (if Ruby installed)**
```bash
cd /scratch/qhuang62/2026-iclr-blogpost-earth4d
bundle install
bundle exec jekyll serve --future --port 8080

# Open in browser: http://localhost:8080/2026/blog/2026/earth4d-world-models/
```

**What to check:**
- [ ] Blog post renders without errors
- [ ] All 5 placeholder images load
- [ ] Citations work (clickable references)
- [ ] Table of contents generates
- [ ] Math equations render ($$...$$)
- [ ] Responsive on mobile (use browser dev tools)

### Step 2: Replace Placeholder Images with Real Ones

The placeholder images are just for testing. You need to replace them with actual figures.

**Where to get real images:**

1. **`earth4d_architecture.png`** - Download from GitHub:
   ```bash
   wget https://raw.githubusercontent.com/legel/deepearth/main/docs/earth4d_spacetime_encoder.png \
        -O /scratch/qhuang62/2026-iclr-blogpost-earth4d/assets/img/2026-04-27-earth4d-world-models/earth4d_architecture.png
   ```

2. **`lfmc_results.png`** - Extract from workshop PDF Figure 3:
   ```bash
   # Page 3 of: /scratch/qhuang62/deepearth-base/encoders/xyzt/deepearth_workshopsubmission.pdf
   # Use a PDF tool to extract/screenshot the figure
   ```

3. **`rgb_reconstruction.png`** - Extract from workshop PDF Figure 4:
   ```bash
   # Page 4 of the workshop PDF
   ```

4. **`collision_heatmap.png`** - Extract from workshop PDF Figure 6 (page 8)
   OR generate from your collision profiling data

5. **`compression_tradeoff.png`** - Already has correct data, just needs better styling

**See `/scratch/qhuang62/deepearth-base/encoders/xyzt/iclr_blog/IMAGE_CHECKLIST.md` for detailed extraction instructions.**

### Step 3: Commit and Push to GitHub

```bash
cd /scratch/qhuang62/2026-iclr-blogpost-earth4d

# Check what files changed
git status

# Add your blog post files
git add _posts/2026-04-27-earth4d-world-models.md
git add assets/bibliography/2026-04-27-earth4d-world-models.bib
git add assets/img/2026-04-27-earth4d-world-models/

# Commit
git commit -m "Add Earth4D blog post: Production-Ready Space-Time Encoding for World Models"

# Push to your fork
git push origin main
```

### Step 4: Create Pull Request to ICLR Repository

1. Go to: https://github.com/qhuang62/2026-iclr-blogpost-earth4d
2. Click "Contribute" → "Open pull request"
3. **IMPORTANT:** PR title MUST be exactly: `2026-04-27-earth4d-world-models`
4. Add description explaining the blog post
5. Submit PR

### Step 5: Wait for GitHub Actions Build

- GitHub Actions will automatically build your blog
- Check the "Actions" tab for build status
- If successful, a bot will comment with a preview URL
- Preview URL will be something like: `https://qhuang62.github.io/2026-iclr-blogpost-earth4d/blog/2026/earth4d-world-models/`

### Step 6: Submit to OpenReview

1. Go to: https://openreview.net/group?id=ICLR.cc/2026/BlogPosts
2. Click "Submit"
3. Fill in:
   - **Title:** Earth4D: Production-Ready Space-Time Positional Encoding for World Models
   - **Blog Post Name:** 2026-04-27-earth4d-world-models
   - **Blog Post URL:** (use the preview URL from GitHub)
   - **Abstract:** (from the blog post front matter)
4. Declare conflicts of interest
5. Submit before **December 7, 2025 23:59 AOE**

---

## 📁 File Locations

### In the Blog Repository (`/scratch/qhuang62/2026-iclr-blogpost-earth4d/`)
- Blog post: `_posts/2026-04-27-earth4d-world-models.md`
- Citations: `assets/bibliography/2026-04-27-earth4d-world-models.bib`
- Images: `assets/img/2026-04-27-earth4d-world-models/*.png`

### Helper Documentation (`/scratch/qhuang62/deepearth-base/encoders/xyzt/iclr_blog/`)
- `README.md` - Overview and status
- `QUICK_START.md` - TL;DR guide
- `SUBMISSION_GUIDE.md` - Detailed step-by-step instructions
- `IMAGE_CHECKLIST.md` - How to get/create real images

---

## 🖼️ Image Status

| Image | Size | Status | Action Needed |
|-------|------|--------|---------------|
| `earth4d_architecture.png` | 24KB | ⚠️ Placeholder | Download from GitHub or PDF Fig 2 |
| `lfmc_results.png` | 21KB | ⚠️ Placeholder | Extract from PDF page 3 (Fig 3) |
| `rgb_reconstruction.png` | 9.9KB | ⚠️ Placeholder | Extract from PDF page 4 (Fig 4) |
| `collision_heatmap.png` | 33KB | ⚠️ Placeholder | Extract from PDF page 8 or generate |
| `compression_tradeoff.png` | 33KB | ⚠️ Placeholder | Regenerate with better styling |

**Total size:** 121KB (well under 500KB limit per image)

---

## ⚠️ Important Notes

### Why I Created Those .md Files

The files in `/scratch/qhuang62/deepearth-base/encoders/xyzt/iclr_blog/` are:
1. **The source content** (blog post + bibliography) - These have been **copied to your repo**
2. **Helper guides** for you to reference during submission
3. **NOT part of the blog itself** - They're documentation for you

The **actual blog** is now in: `/scratch/qhuang62/2026-iclr-blogpost-earth4d/`

### Testing is Critical

**DO NOT skip local testing!** Testing locally catches:
- Jekyll build errors (YAML syntax, missing files)
- Broken image links
- Citation errors
- Mobile responsive issues

If you skip testing and just create the PR, you might discover errors only after waiting for GitHub Actions to build (which takes 5-10 minutes each time).

### Placeholder Images are OK for Initial Testing

You can test the blog locally RIGHT NOW with the placeholder images. This lets you:
- Verify the blog post renders correctly
- Check that citations work
- See the layout and formatting
- Fix any YAML or Jekyll errors

Once testing looks good, then replace with real images before final submission.

---

## 🚀 Quick Commands

### Test Locally Right Now
```bash
cd /scratch/qhuang62/2026-iclr-blogpost-earth4d
docker-compose up
# Open: http://localhost:8080/2026/blog/2026/earth4d-world-models/
```

### Download Architecture Diagram
```bash
cd /scratch/qhuang62/2026-iclr-blogpost-earth4d/assets/img/2026-04-27-earth4d-world-models/
wget https://raw.githubusercontent.com/legel/deepearth/main/docs/earth4d_spacetime_encoder.png \
     -O earth4d_architecture.png
```

### Check Git Status
```bash
cd /scratch/qhuang62/2026-iclr-blogpost-earth4d
git status
git diff _posts/2026-04-27-earth4d-world-models.md
```

---

## ✅ Ready to Test!

Your blog is now set up and ready to test locally. The blog post is complete with:
- ✅ 4500 words of content
- ✅ 9 sections (Introduction → Conclusion)
- ✅ 11 citations
- ✅ 5 placeholder images (for testing)
- ✅ Proper YAML front matter
- ✅ Table of contents

**Next action:** Run `docker-compose up` and check http://localhost:8080/2026/blog/2026/earth4d-world-models/

Then replace placeholder images with real ones from your workshop PDF and DeepEarth GitHub.

---

**Questions?** See the detailed guides in `/scratch/qhuang62/deepearth-base/encoders/xyzt/iclr_blog/`
