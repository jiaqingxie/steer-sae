# Adapted from: https://github.com/QwenLM/Qwen2.5-Math/blob/main/evaluation/examples.py
import json

def get_examples(dataset):
    examples = {}
    if dataset in ["gsm8k", "svamp", "mawps", "asdiv", "gsm8k_train"]:
        examples[dataset] = [
            (
                "There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?",
                "There are 15 trees originally. Then there were 21 trees after some more were planted. So there must have been 21 - 15 = 6. ",
                6,
            ),
            (
                "If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?",
                "There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5. ",
                5,
            ),
            (
                "Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?",
                "Originally, Leah had 32 chocolates. Her sister had 42. So in total they had 32 + 42 = 74. After eating 35, they had 74 - 35 = 39. ",
                39,
            ),
            (
                "Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?",
                "Jason started with 20 lollipops. Then he had 12 after giving some to Denny. So he gave Denny 20 - 12 = 8. ",
                8,
            ),
            (
                "Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?",
                "Shawn started with 5 toys. If he got 2 toys each from his mom and dad, then that is 4 more toys. 5 + 4 = 9. ",
                9,
            ),
            (
                "There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?",
                "There were originally 9 computers. For each of 4 days, 5 more computers were added. So 5 * 4 = 20 computers were added. 9 + 20 is 29. ",
                29,
            ),
            (
                "Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?",
                "Michael started with 58 golf balls. After losing 23 on tuesday, he had 58 - 23 = 35. After losing 2 more, he had 35 - 2 = 33 golf balls. ",
                33,
            ),
            (
                "Olivia has $23. She bought five bagels for $3 each. How much money does she have left?",
                "Olivia had 23 dollars. 5 bagels for 3 dollars each will be 5 x 3 = 15 dollars. So she has 23 - 15 dollars left. 23 - 15 is 8. ",
                8,
            ),

        ]

    elif dataset == "aqua":
        examples[dataset] = [
            (
                "Answer Choices: (A) 50 (B) 45 (C) 65 (D) 78 (E) 64. John found that the average of 15 numbers is 40. If 10 is added to each number then the mean of the numbers is?",
                "If 10 is added to each number, then the mean of the numbers also increases by 10. So the new mean would be 50. ",
                "(A)",
            ),
            (
                "Answer Choices: (A) 1/2 (B) 3/2 (C) 5/2 (D) 4/2 (E) 7/2. If a / b = 3/4 and 8a + 5b = 22,then find the value of a.",
                "a / b = 3/4, then b = 4a / 3. So 8a + 5(4a / 3) = 22. This simplifies to 8a + 20a / 3 = 22, which means 44a / 3 = 22. So a is equal to 3/2. ",
                "(B)",
            ),
            (
                "Answer Choices: (A) 53 km (B) 55 km (C) 52 km (D) 60 km (E) 50 km. A person is traveling at 20 km/hr and reached his destiny in 2.5 hr then find the distance?",
                "The distance that the person traveled would have been 20 km/hr * 2.5 hrs = 50 km. ",
                "(E)",
            ),
            (
                "Answer Choices: (A) 1156 (B) 1392 (C) 1480 (D) 1562 (E) 1788. How many keystrokes are needed to type the numbers from 1 to 500?",
                "There are 9 one-digit numbers from 1 to 9. There are 90 two-digit numbers from 10 to 99. There are 401 three-digit numbers from 100 to 500. 9 + 90(2) + 401(3) = 1392. ",
                "(B)",
            ),
        ]
    elif dataset == "math":
        examples[dataset] = [
            (
                "Kevin Kangaroo begins hopping on a number line at 0. He wants to get to 1, but he can hop only $\\frac{1}{3}$ of the distance. Each hop tires him out so that he continues to hop $\\frac{1}{3}$ of the remaining distance. How far has he hopped after five hops? Express your answer as a common fraction.",
                "Let's think step by step\nKevin hops $1/3$ of the remaining distance with every hop.\nHis first hop takes $1/3$ closer.\nFor his second hop, he has $2/3$ left to travel, so he hops forward $(2/3)(1/3)$.\nFor his third hop, he has $(2/3)^2$ left to travel, so he hops forward $(2/3)^2(1/3)$.\nIn general, Kevin hops forward $(2/3)^{k-1}(1/3)$ on his $k$th hop.\nWe want to find how far he has hopped after five hops.\nThis is a finite geometric series with first term $1/3$, common ratio $2/3$, and five terms.\nThus, Kevin has hopped $\\frac{\\frac{1}{3}\\left(1-\\left(\\frac{2}{3}\\right)^5\\right)}{1-\\frac{2}{3}} = \\boxed{\\frac{211}{243}}$.",
                "\\frac{211}{243}",
            ),
            (
                "What is the area of the region defined by the equation $x^2+y^2 - 7 = 4y-14x+3$?",
                "Let's think step by step\nWe rewrite the equation as $x^2 + 14x + y^2 - 4y = 10$ and then complete the square,\nresulting in  $(x+7)^2-49 + (y-2)^2-4=10$,\nor $(x+7)^2+(y-2)^2=63$.\nThis is the equation of a circle with center $(-7, 2)$ and radius $\\sqrt{63},$\nso the area of this region is $\\pi r^2 = \\boxed{63\\pi}$.",
                "63\\pi",
            ),
            (
                "If $x^2+y^2=1$, what is the largest possible value of $|x|+|y|$?",
                "Let's think step by step\nIf $(x,y)$ lies on the circle,\nso does $(x,-y),$ $(-x,-y),$ and $(-x,-y),$ (which all give the same value of $|x| + |y|$),\nso we can assume that $x \\ge 0$ and $y \\ge 0.$\nThen $|x| + |y| = x + y.$  Squaring, we get\n\\[(x + y)^2 = x^2 + 2xy + y^2 = 1 + 2xy.\\]\nNote that $(x - y)^2 \\ge 0.$\nExpanding, we get $x^2 - 2xy + y^2 \\ge 0,$ so $2xy \\le x^2 + y^2 = 1.$\nHence,\\[1 + 2xy \\le 2,\\]which means $x + y \\le \\sqrt{2}.$\nEquality occurs when $x = y = \\frac{1}{\\sqrt{2}},$\nso the maximum value of $|x| + |y|$ is $\\boxed{\\sqrt{2}}.$",
                "\\sqrt{2}",
            ),
            (
                "If $f(x)=\\frac{ax+b}{cx+d}, abcd\\not=0$ and $f(f(x))=x$ for all $x$ in the domain of $f$, what is the value of $a+d$?",
                "Let's think step by step\nThe condition $f(f(x))$ means that $f$ is the inverse of itself,\nso its graph is symmetrical about the line $y = x$.\nWith a rational function of this form, we will have two asymptotes:\na vertical one at $x=-d/c$ if $cx+d$ does not divide $ax+b$,\nand a horizontal one at $y=a/c$,\nif we take the limit of $f(x)$ as $x$ goes to $\\pm\\infty$.\nIn order for $f$ to be its own inverse, the intersection of the asymptotes must lie on the line $y=x$\nso that it and its asymptotes reflect onto themselves.\nThis means that $-d/c=a/c$,\nand therefore $-d=a$ and $a+d=\\boxed{0}$.",
                "0",
            ),
            (
                "Expand $(2z^2 + 5z - 6)(3z^3 - 2z + 1)$.",
                "Let's think step by step\n$$\\begin{array}{crrrrrrr}\n& & & 3z^3 & & -2z & + 1 & \\\\\n\\times & & & & 2z^2 & +5z & -6 \\\\\n\\cline{1-7}\\rule{0pt}{0.17in}\n& & & -18z^3 & & +12z & -6 & \\\\\n& & +15z^4 & & -10z^2 & +5z & & \\\\\n+ & 6z^5 & & -4z^3 & +2z^2 & & & \\\\\n\\cline{1-7}\\rule{0pt}{0.17in}\n& 6z^5 & +15z^4 & -22z^3 & - 8z^2 &+17z & -6 &\n\\end{array}$$",
                "6z^5+15z^4-22z^3-8z^2+17z-6",
            ),
        ]
    return examples