# Predicting Professor Ratings from Review Text 🎓

Ever wondered if a computer could read student reviews and guess the star rating? That's exactly what I tried to figure out in this project!

## What's This About?

I built a machine learning model that reads professor reviews (just the text) and predicts what star rating the student gave. Think of it like teaching a computer to understand if a review is positive or negative, but with 5 different levels instead of just two.

**The cool part?** After some experimentation with handling imbalanced data, the model got really good at this - **92.6% accuracy**!

## The Challenge

Here's what made this interesting:
- Students write reviews very differently
- Sarcasm exists ("Oh great, another impossible exam...")
- Mixed feelings are common ("Great professor, terrible exams")
- The subtle difference between a 3-star and 4-star review is hard even for humans

## What I Found

I ran three different experiments to see what works best:

| Experiment | Accuracy | What I Changed |
|------------|----------|----------------|
| First try | 74.8% | Just basic training |
| Second try | 81.6% | Trained it longer |
| **Final version** | **92.6%** | Fixed the imbalance problem |

The biggest breakthrough? Handling the fact that 62% of all reviews are 5-stars. Without fixing this, the model basically just learned to guess "5 stars" for everything!

### Before and After

Here's how much better the model got at predicting each rating:

| Star Rating | Before | After |
|-------------|--------|-------|
| ⭐ (1-star) | 65% correct | **100% correct** |
| ⭐⭐ (2-star) | 35% correct | **100% correct** |
| ⭐⭐⭐ (3-star) | 11% correct | **96% correct** |
| ⭐⭐⭐⭐ (4-star) | 8% correct | **71% correct** |
| ⭐⭐⭐⭐⭐ (5-star) | 93% correct | 95% correct |

The improvement in 3-star and 4-star predictions is huge!

## How It Works

I used something called **transfer learning** with a model called DistilBERT. Instead of training from scratch (which would take weeks and tons of data), I started with a model that already understands English from reading billions of words. Then I just taught it: "this type of language = this star rating."

### The Secret Sauce

The data was super imbalanced - way more 5-star reviews than anything else. To fix this, I made the model "care more" about getting the rare ratings right. It's like telling it: "Hey, if you mess up a 2-star review, that's a much bigger deal than messing up a 5-star review."

This one change improved accuracy by almost 18 percentage points!

## The Data

I collected 818 reviews from 5 popular UMD CS professors using the Planet Terp API:
- Justin Wyss-Gallifent (319 reviews)
- Nelson Padua-Perez (241 reviews)
- Clyde Kruskal (100 reviews)
- Elias Gonzalez (88 reviews)
- Anwar Mamat (70 reviews)

Most reviews are around 100 words, but some students really had a lot to say (one review was 1,127 words!).

## Files in This Repo
```
├── project.ipynb       # All the code and analysis
├── project.pdf         # Presentation slides
└── README.md          # You're reading it!
```

## Running It Yourself

If you want to try this out:

1. Clone this repo
2. Install the packages (you'll need PyTorch, transformers, pandas, numpy, scikit-learn, matplotlib, seaborn)
3. Open `project.ipynb` and run the cells!

Just a heads up - training takes a while even on a good GPU, so grab some coffee ☕

## What I Learned

**The big lessons:**
1. Transfer learning is incredibly powerful - you don't always need millions of data points
2. In imbalanced datasets, looking at overall accuracy can be super misleading
3. Sometimes a simple fix (class weighting) works way better than complex solutions
4. Training longer helps, but fixing fundamental problems helps even more

## What Could Be Better

This project has some limitations:
- Only 818 reviews (not a huge dataset)
- Just CS professors (other departments might be different)
- The 5-star bias is real - students really love leaving 5-star reviews!

## Ideas for Future Work

If I had more time, here's what I'd try:
- Collect data from professors across all departments
- Try bigger models like full BERT or RoBERTa
- Analyze which specific words predict each rating
- Build a website where you can paste a review and see the prediction in real-time
- Add data augmentation to create more examples of rare ratings

## Why This Matters

This kind of analysis could actually be useful:
- Students could quickly understand overall professor sentiment
- Departments could spot concerning patterns in feedback
- Researchers could study educational quality at scale
- It's a stepping stone for more sophisticated educational analytics

## About This Project

I built this for a computer science course at the University of Maryland. It combines natural language processing, machine learning, and real-world data to solve an actually useful problem.

Made by **Harsh Bhati** 🐢  
University of Maryland, College Park

---

*P.S. - If you're one of the professors whose reviews I used for this: thank you for being awesome teachers! The high number of 5-star reviews shows how much students appreciate you.* 😊
