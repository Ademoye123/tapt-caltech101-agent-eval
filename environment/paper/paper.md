# TAPT: Test-Time Adversarial Prompt Tuning for Robust Inference in Vision-Language Models

CVPR 2025 · arXiv:2411.13136 · (plain-text render of paper.pdf)

                                                  TAPT: Test-Time Adversarial Prompt Tuning for Robust Inference in
                                                                     Vision-Language Models

                                                                Xin Wang1 , Kai Chen1 , Jiaming Zhang2 , Jingjing Chen1 , Xingjun Ma1 *
                                                         1
                                                             Shanghai Key Lab of Intell. Info. Processing, School of CS, Fudan University
                                                                        2
                                                                          Hong Kong University of Science and Technology




arXiv:2411.13136v1 [cs.CV] 20 Nov 2024
                                                                                                                             a photo of a <cat>               a photo of a <cat>
                                                                     Abstract                                                                       False                             False

                                                                                                                                  𝑣! , ⋯ 𝑣" <cat>                  𝑣! , ⋯ 𝑣" <cat>
                                                                                                                                                    False                             True

                                         Large pre-trained Vision-Language Models (VLMs) such                             𝑣! 𝑥# , ⋯ 𝑣" 𝑥#   <cat>
                                                                                                                                                    True
                                                                                                                                                            𝑣! 𝑥$ , ⋯ 𝑣" 𝑥$   <cat>
                                                                                                                                                                                      True
                                         as CLIP have demonstrated excellent zero-shot general-
                                         izability across various downstream tasks. However, re-             Figure 1. Inference with different prompts. Top: inference with
                                         cent studies have shown that the inference performance of           hand-crafted prompts fails to recognize the class ‘cat’; Middle:
                                         CLIP can be greatly degraded by small adversarial pertur-           Inference with fixed prompts tuned by APT methods cannot rec-
                                         bations, especially its visual modality, posing significant         ognize all adversarial images; Bottom: Inference with test-time
                                         safety threats. To mitigate this vulnerability, in this pa-         prompts optimized for each image produces accurate recognitions.
                                         per, we propose a novel defense method called Test-Time
                                         Adversarial Prompt Tuning (TAPT) to enhance the infer-
                                         ence robustness of CLIP against visual adversarial attacks.         risks across a wide range of downstream applications.
                                         TAPT is a test-time defense method that learns defensive                Adversarial training [12, 25, 51, 56] is a general defense
                                         bimodal (textual and visual) prompts to robustify the in-           strategy that augments training data with adversarial ex-
                                         ference process of CLIP. Specifically, it is an unsupervised        amples crafted to mislead the model. While this method
                                         method that optimizes the defensive prompts for each test           has proven effective [8], it entails costly min-max train-
                                         sample by minimizing a multi-view entropy and aligning              ing, which restricts its practicality—especially for large
                                         adversarial-clean distributions. We evaluate the effective-         VLMs, where standard training alone can cost millions of
                                         ness of TAPT on 11 benchmark datasets, including Ima-               dollars [37]. To address the efficiency limitations of ad-
                                         geNet and 10 other zero-shot datasets, demonstrating that           versarial training, recent research has introduced adversar-
                                         it enhances the zero-shot adversarial robustness of the orig-       ial prompt tuning (APT) methods [22, 58, 65], which align
                                         inal CLIP by at least 48.9% against AutoAttack (AA), while          learnable text prompts with adversarial image embeddings.
                                         largely maintaining performance on clean examples. More-            These differentiable prompts [18, 62, 63] provide a more
                                         over, TAPT outperforms existing adversarial prompt tuning           cost-effective alternative to adversarial training.
                                         methods across various backbones, achieving an average                  Despite its promising results, APT faces three funda-
                                         robustness improvement of at least 36.6%.                           mental limitations: (1) The distribution-dependent nature
                                                                                                             of learnable prompts restricts their generalization to out-of-
                                                                                                             distribution scenarios and novel tasks. (2) The requirement
                                                                                                             for task-specific annotated data presents significant chal-
                                         1. Introduction                                                     lenges for zero-shot applications. (3) While APT meth-
                                         Vision-Language Models (VLMs) pre-trained on large-                 ods improve robustness on specific tasks [41], they of-
                                         scale datasets of image-text pairs have emerged as powerful         ten compromise the model’s overall generalization perfor-
                                         backbones for numerous applications, including computer             mance [32, 42]. Given that real-world applications in-
                                         vision [16, 33, 55], medical image analysis [15, 50], and           volve an unbounded set of potential tasks, evaluating ad-
                                         robotics [3, 17, 38]. Despite these advancements, studies           versarial robustness solely on predefined downstream tasks
                                         indicate that even minor adversarial perturbations in input         proves insufficient. Therefore, achieving zero-shot adver-
                                         images can significantly degrade the inference performance          sarial robustness—maintaining performance against adver-
                                         of VLMs [9, 25, 43, 57, 61], thereby posing critical safety         sarial attacks on unseen tasks without task-specific train-
                                                                                                             ing—remains an open challenge.
                                           * Corresponding author.                                               The key to addressing the zero-shot adversarial robust-


                                                                                                         1
ness challenge lies in adaptively identifying a robust prompt         robustness against AutoAttack by 36.6% with ViT-B/16,
that consistently aligns an adversarial image embedding               and by 38.0% with ViT-B/32, respectively.
with its correct text embedding for each test sample. To this
end, we propose a simple yet effective framework called             2. Related Work
Test-Time Adversarial Prompt Tuning (TAPT), which
dynamically tunes a robust prompt on the fly based on only          Here, we briefly review related works on adversarial attacks
the provided test sample. Specifically, TAPT learns defen-          and defenses for pre-trained VLMs, and test-time adapta-
sive prompts by minimizing two unsupervised losses: (1)             tion techniques proposed to improve generalization.
multi-view entropy, which ensures consistent predictions            Adversarial Attacks on Pre-trained VLMs Adversar-
across various augmented views of each test sample, and (2)         ial attacks on pre-trained VLMs are broadly categorized
adversarial-clean embedding alignment, which aligns the             as white-box or black-box attacks according to the threat
means and variances of test sample embeddings with pre-             model. In white-box attacks, the attacker has full access to
computed adversarial-clean embeddings of a public dataset           the model parameters and can directly compute adversarial
(e.g., ImageNet) to enhance inference robustness. Notably,          gradients [24, 25, 57, 66]. Black-box attacks, on the other
TAPT operates during the inference phase without requiring          hand, restrict the attacker to querying model outputs [10,
any task-specific training set or annotations. The different        13, 23, 46, 53, 54, 60, 61]. Traditional single-modal attacks
inference schemes of training-time defense APT methods              designed for vision models, such as PGD [25], DI [53], and
and our test-time defense TAPT are illustrated in Figure 1.         AutoAttack [8], can be used directly to attack the image
    We evaluated TAPT against both white-box and black-             encoders of pre-trained VLMs. Several recent multi-modal
box adversarial attacks across 11 benchmark datasets.               attacks have targeted pre-trained VLMs by simultaneously
TAPT demonstrated superior performance compared to                  exploiting vulnerabilities in both their image and text en-
vanilla CLIP (using hand-crafted prompts) and existing              coders. For example, Co-Attack [57] pioneered white-box
APT methods across three different prompt designs: visual-          multi-modal attacks, perturbing both image and text modal-
only prompts, visual-language (V-L) joint prompts, and V-L          ities concurrently. SGA [23] extended Co-Attack to the
independent prompts. By dynamically adapting prompts at             black-box setting, improving the transferability of multi-
inference time, TAPT opens a new direction for safeguard-           modal adversarial examples. VLATTACK [54] further gen-
ing the inference process of pre-trained VLMs. Offering a           erates adversarial examples by fusing perturbations of im-
flexible and efficient solution for zero-shot adversarial ro-       ages and texts from both single-modal and multi-modal.
bustness while maintaining performance on clean samples,            Adversarial Defenses for Pre-trained VLMs Adversar-
our TAPT method bridges the gap between the need for ad-            ial training/tuning is a widely used defense strategy for
versarial robustness and the performance challenges posed           pre-trained VLMs, with existing methods generally clas-
by open, real-world environments.                                   sified into adversarial contrastive tuning [27, 35, 48, 49,
    In summary, our main contributions are:                         51, 64] and adversarial prompt tuning [22, 58, 65]. Ad-
• We propose a novel test-time defense method named                 versarial contrastive tuning focuses on enhancing the ad-
   Test-Time Adversarial Prompt Tuning (TAPT) to ro-                versarial robustness of the backbone model. For example,
   bustify the zero-shot inference process of pre-trained           TeCoA [27] examines the impact of fine-tuning and visual
   VLMs. TAPT adapts the prompt for each test image                 prompt tuning on the zero-shot adversarial robustness of
   to achieve robust inference without the need for task-           VLMs. FARE [35] improves CLIP’s image encoder through
   specific tuning for downstream datasets. To the best of          unsupervised adversarial fine-tuning, enhancing adversar-
   our knowledge, TAPT is the first inference-time adversar-        ial robustness in models like LLaVA and OpenFlamingo
   ial defense method for pre-trained VLMs.                         without retraining. PMG-AFT [48] introduces an auxil-
• TAPT introduces an adversarial-clean alignment loss that          iary branch to boost zero-shot adversarial robustness, while
   aligns the distribution of a test sample with pre-computed       MMCoA [64] investigates VLM vulnerabilities to multi-
   adversarial-clean distributions from a public dataset (Im-       modal attacks. In contrast, AdvPT [58] and APT [22] offer
   ageNet), thereby improving adversarial robustness while          an efficient approach to bolster the adversarial robustness
   maintaining accuracy on clean samples. It also leverages         of VLMs by tuning only the textual prompts without alter-
   the advantage of APT by using pre-tuned prompts on Ima-          ing the model parameters. FAP [65] further refines APT
   geNet to further enhance zero-shot adversarial robustness.       by balancing cross-modal consistency between benign and
• We conduct extensive experiments on 11 datasets, includ-          adversarial inputs. While these methods are all training-
   ing ImageNet and 10 other zero-shot datasets. Our results        time defense methods that need to pre-tune the prompt for a
   demonstrate that TAPT significantly outperforms existing         specific downstream task, in this work we propose a novel
   APT baselines against both white-box and black-box at-           test-time defense method that optimizes the prompt on the
   tacks. Specifically, TAPT improves zero-shot adversarial         fly during inference and is task-agnostic.

                                                                2
              (a) CLIP                             (b) Visual Prompt                         (c) V-L Joint Prompt                  (d) V-L Independent Prompt

          Maximize Similarity                      Maximize Similarity                          Maximize Similarity                       Maximize Similarity



  Image Encoder         Text Encoder       Image Encoder         Text Encoder           Image Encoder         Text Encoder        Image Encoder         Text Encoder



      ⋯                         ⋯              ⋯        🔥                ⋯                  ⋯        🔥 🔥              ⋯               ⋯        🔥 🔥              ⋯


                     a photo of a <cat>                       a photo of a <cat>                           a photo of a <cat>                        a photo of a <cat>


 adversarial image                        adversarial image                            adversarial image                         adversarial image



Figure 2. An illustration of CLIP and different adversarial prompt tuning schemes. (a) The original CLIP [33]; (b) - (d) Adversarial prompt
tuning with three distinctive prompt designs: Visual Prompt (b), V-L Joint Prompt (c), and V-L Independent Prompt (d).


Test-Time Adaptation Test-time adaptation (TTA) meth-                                  terized by θI , and the text encoder as T , parameterized by
ods enhance the generalization performance of pre-trained                              θT . Considering a K-class classification problem, where
models by adapting to individual test samples or batches,                              each image x is associated with a class label in the for-
addressing distribution shifts between training (source) and                           mat "a photo of a <class>". For a clean sample
testing (target) data. Early TTA methods [28, 36] address                              x ∈ [0, 1]d and a target CLIP model comprising {I, T }, a
domain shift by updating batch normalization statistics                                white-box adversarial attack seeks to generate an adversar-
based on test batch statistics. Building on this, TENT [45]                            ial example x′ that maximizes the model loss as follows:
improves adaptation by updating batch normalization layers
to minimize the entropy of predicted probabilities for each                                                 x′ = arg max L(I(x′ ), T (y)),                             (1)
test batch, while MEMO [59] extends this by minimizing                                                             ∥x′ −x∥∞ ≤ϵ
entropy over multiple augmented input samples. Inspired
by TENT, subsequent TTA methods such as CoTTA [47]                                     where L(·) represents the loss function, x′ is the adversarial
and EATA [30] enable pre-trained models to adapt to con-                               example, and ϵ denotes the perturbation budget.
tinuously shifting test distributions. More recently, methods
like TPT [39] and PromptAlign [1] have focused on tuning                               Adversarial Prompt Tuning (APT) APT applies ad-
prompts exclusively at test time to ensure consistent predic-                          versarial training during the prompt tuning process to
tions across various augmented views of a test sample. Kim                             enhance adversarial robustness, with recent APT meth-
et al. [19] finds that combining TTA with AGL [4] further                              ods primarily developed to defend the visual component
improves out-of-distribution (OOD) accuracy prediction. In                             of CLIP. Rather than relying on hand-crafted prompts
contrast to existing TTA methods which primarily empha-                                like "a photo of a <class>", APT learns robust
size performance on clean samples, in this work we intro-                              prompts from training data to improve adversarial robust-
duce TAPT, a test-time defense, that specifically enhances                             ness on downstream tasks. As shown in Figure 2, APT
zero-shot adversarial robustness against potential attacks.                            can be extended to three distinctive prompt designs: (1)
                                                                                       Visual-only (prompts applied solely to the vision branch),
                                                                                       (2) Vision-Language (V-L) joint (shared prompts across
3. Proposed Method                                                                     both branches), and (3) V-L independent (separate prompts
3.1. Preliminaries                                                                     for each branch). Specifically, APT optimizes the prompts
                                                                                       P = {Pv , Pt } ∈ RL×D , where Pv and Pt are the visual
Threat Model We assume a white-box threat model in                                     and textual prompts, respectively, L denotes the number of
which the adversary has full knowledge of the target                                   prompt tokens, and D is the embedding dimension.
model’s architecture and parameters, allowing direct per-                                 Given a downstream training dataset Dtrain = {(x, y)},
turbation of test images based on adversarial gradients prior                          the learnable visual prompt Pv is appended to the vi-
to inference. The defender, or model owner, can deploy any                             sual input tokens, forming the sequence {x, Pv } =
defense strategies to protect against potential adversarial at-                        {CLS, e1 , e2 , · · · , eM , Pv }. Similarly, the textual prompt
tacks. Specifically, we focus on securing CLIP zero-shot                               Pt is appended to the text input, forming {y, Pt }. APT en-
inference, where the defender lacks access to task-specific                            hances adversarial robustness by aligning the embeddings
training data or annotations of the downstream application.                            of clean text with those of adversarial images through a min-
CLIP We denote the CLIP image encoder as I, parame-                                    max optimization. The corresponding optimization prob-


                                                                                   3
                                                       Adversarial Alignment

                                   ℒ()*+"                    ℒ()*+"                   ℒ()*+"
                                                                                                                                                                                          a photo of a <class 1>



                                       Image Encoder              Image Encoder           Image Encoder               Text Encoder                  Text Encoder       Text Encoder
                               ⋯                            ⋯                         ⋯                                              ⋯                             ⋯                  ⋯   a photo of a <class 2>
                       ⋯                                                          ⋯                           ×                          ⋯
                                                                                                                                                                                                    ⋯
                                                                                                          ℒ!"#$%&'
 adversarial image                 🔥                         🔥                        🔥                                              🔥                             🔥                  🔥   a photo of a <class K>
                                   ℒ()*+"                    ℒ()*+"                   ℒ()*+"
                                                       Clean Alignment



Figure 3. An overview of our proposed TAPT method: Given an adversarial image, TAPT generates multiple augmented views of the
image and retains only those views with low entropy in their averaged prediction probabilities. During inference, TAPT then optimizes
the prompt by minimizing multi-view entropy across these selected views while aligning their embedding distribution with pre-computed
adversarial-clean statistics from a public dataset (ImageNet).


Algorithm 1 Test-Time Adversarial Prompt Tuning                                                               3.2. Test-Time Adversarial Prompt Tuning (TAPT)
 1: Input: input image x, image encoder I, text encoder                                                       Framework Overview As illustrated in Figure 3, TAPT
    T , augmentation function A, entropy threshold τ , pre-                                                   comprises two main modules: (1) multi-view entropy-
    computed Dpublic statistics {µadv , σadv , µclean , σclean }                                              based sample selection, and (2) adversarial-clean embed-
 2: Output: Learnable prompts P                                                                               ding alignment. The defense procedure of TAPT op-
 3: 1. Initialize adversarial prompts:                                                                        erates as follows. Given a test image x ∈ Dtest ,
 4:    P ← APT(Dpublic , ϵ; I, T )                                                                            TAPT begins by generating M randomly augmented views
 5: 2. Multi-view entropy based sample selection:                                                             {A1 (x), A2 (x), . . . , AM (x)} through random augmenta-
 6:    Select the top τ entropy from A(x) to form Hτ (x).                                                     tions A. The multi-view entropy-based sample selection
 7: 4. Compute multi-view entropy loss:                                                                       module then chooses the top-K views with the lowest en-
 8:    Calculate Lentropy over selected views Hτ (x)                                                          tropy in their averaged prediction probabilities. Using
 9: 5. Compute current embedding statistics:                                                                  these selected views, TAPT optimizes the prompt P dur-
10:    for each layer l in I do                                                                               ing inference by minimizing multi-view entropy and en-
11:      µl = mean(Il (x̂, P )) for x̂ in Hτ (x)                                                              forcing adversarial-clean alignment. The prompt is reset
12:      σl2 = std(Il (x̂, P )) for x̂ in Hτ (x)                                                              to its initial state before processing each new test sample
13:    end for                                                                                                or batch. The complete procedure of TAPT is outlined in
14: 6. Adversarial-clean embedding alignment:                                                                 Algorithm 1.
                1 PL                                       
15:    Ladv =          ∥µl − µadv,l ∥1 + σl2 − σadv,l
                                                   2                                                          Multi-View Entropy-Based Sample Selection Following
               L l=1                                      1
                                                                                                              prior works [1, 39], we first discard ineffective augmented
                 1 PL                                           
16:    Lclean =         ∥µl − µclean,l ∥1 + σl2 − σclean,l
                                                      2                                                       views A(x) (e.g., when a random crop removes essen-
                 L l=1                                         1
                                                                                                              tial image content) by applying a selection filter with a
17: 7. Optimize prompts:
                                                                                                              threshold τ . This filter retains only augmented views with
18:    LTAPT = Lentropy + αLadv + (1 − α)Lclean                                                               low entropy (high-confidence predictions). Specifically, we
19:    Optimize P ← Minimize LTAPT                                                                            define the set of selected augmented views as Hτ (x) =
                                                                                                              {Aj (x)|1 ≤ j ≤ M, H(Aj (x)) ≤ Hτ }, where Hτ is the
                                                                                                              entropy threshold corresponding to the top τ lowest entropy
                                                                                                              values among all M augmentations. TAPT then optimizes
lem can be formulated as:
                                                                                                              prompts by minimizing the multi-view entropy of the aver-
                                                                                                              aged prediction probability over these selected views:
                                                        ′
   arg min EDtrain      max        L(I(x , Pv ), T (y, Pt )),                             (2)
       P             ∥x′ −x∥∞ ≤ϵ                                                                                                         K
                                                                                                                                         X
                                                                                                                  Lentropy = −                     p̃(yi |Hτ (x), P ) log p̃(yi |Hτ (x), P ), (3)
             ′
                                                                                                                                             i=1
where I(x , Pv ) and T (y, Pt ) denote the adversarial image
                                                                                                              where p̃(yi |Hτ (x), P) = |Hτ1(x)| x̂∈Hτ (x) p(yi |x̂, P) de-
                                                                                                                                                P
embedding and text embedding, respectively. APT methods
tune the prompts on the training dataset of each downstream                                                   notes the average predicted probability for class yi over the
task. At test time, the tuned prompts are fixed to perform                                                    selected augmented views Hτ (x) when using prompt P .
inference for different test images.                                                                          This optimization encourages the model to achieve consis-


                                                                                                          4
tent predictions by adjusting the prompt specifically for the             4. Experiments
given instance.
                                                                          4.1. Experimental Setup
Adversarial-Clean Embedding Alignment An adversarial
test image x can shift the image embeddings produced by                   Datasets and Models We experiment on 11 benchmark
the image encoder I(x, P ) compared to those from clean                   datasets (ImageNet test set and 10 other zero-shot test
images, potentially misleading the model. To mitigate this                datasets): ImageNet [34], Caltech101 [11], DTD [7], Eu-
vulnerability, we align the mean and variance of the test                 roSAT [14], Pets [31], Aircraft [26], Food101 [6], Flow-
image embedding with pre-computed statistics from a pub-                  ers [29], Cars [21], SUN397 [52], and UCF101 [40]. Our
lic dataset Dpublic . Ideally, this public dataset would orig-            experiments focus on the CLIP model, specifically utiliz-
inate from the original CLIP pre-training data. However,                  ing the ViT-B/16 and ViT-B/32 architectures. Following
since this data is unavailable, we use ImageNet as a proxy,               standard CLIP usage, we used hand-crafted prompts as tex-
given CLIP’s strong zero-shot performance on ImageNet                     tual inputs. For example, the prompt "a photo of a
with standard prompt tuning [5]. Specifically, we compute                 <class>, a type of pet" was applied for the Pets
the mean and variance of the current embeddings for align-                dataset. A summary of these datasets and their correspond-
ment as follows:                                                          ing hand-crafted prompts are provided in the Appendix.
                         1        X
                                                                          Attack Configuration We evaluate the zero-shot adversar-
 µl (Hτ (x); P ) =                         Il (x̂, P ),         (4)
                     |Hτ (x)|                                             ial robustness of CLIP against both white-box and black-
                               x̂∈Hτ (x)
                     P                                            2
                                                                          box attacks. Specifically, we employ PGD-100 [25] for
  2                     x̂∈Hτ (x) (Il (x̂, P ) − µl (Hτ (x); P ))         white-box attacks, DI [53] for black-box attacks, and the
σl (Hτ (x); P ) =
                                     |Hτ (x)| − 1                         more powerful AutoAttack [8]. The hyperparameters for
                                                                (5)       PGD-100 and DI were configured based on the TorchAt-
where Il (x̂, P ) represents the embedding vector at layer                tacks library [20]. Consistent with [27], we use perturbation
l for the augmented input x̂ ∈ Hτ (x) given prompt P .                    budgets of ϵ = 1/255, 2/255, and 4/255 for both attacks.
Here, µl (Hτ (x); P ) and σl2 (Hτ (x); P ) denote the mean                Defense Configuration For existing APT methods, we use
and variance of the test sample embeddings at layer l, re-                the original configurations of APT methods [22, 58] and
spectively. We similarly pre-compute the mean and vari-                   generate adversarial examples using PGD-2 attack with step
ance of embeddings from the public dataset using the robust               size α = 1/255. We then develop more robust versions of
prompt Padv (obtained via APT on the public data) and                     APT methods with different prompt designs, including 1)
clean prompt Pclean (obtained via standard prompt tuning                  adversarial visual prompt tuning (APT-V), 2) adversarial V-
on the public data). These offline statistics are represented             L joint prompt tuning (APT-VLJ), and 3) adversarial V-L
           2                   2
by µadv , σadv , µclean , and σclean , respectively. We then align        independent prompt tuning (APT-VLI).
the mean and variance of the current embeddings with these                Implementation Details For our TAPT method, we initial-
pre-computed statistics as follows:                                       ize the defensive prompt using APT on ImageNet, training
             1X
                 L                                                        for 100 epochs with a batch size of 32 and a learning rate
    Ladv =     (∥µl − µadv,l ∥1 + ∥σl2 − σadv,l
                                          2
                                                ∥1 ), (6)                 of 0.035. To generate augmented views for test-time fine-
             L
                l=1                                                       tuning, we create 63 variations of each test sample using
              L                                                           random resized crops and horizontal flips, resulting in 64
            1X
   Lclean =     (∥µl − µclean,l ∥1 + ∥σl2 − σclean,l
                                             2
                                                     ∥1 ),                images per sample, including the original. From these 64
            L
                l=1                                                       images, we select the top 10% most confident predictions
                                                               (7)        (with the lowest entropy) and compute the average entropy
  LTAPT = Lentropy + αLadv + (1 − α)Lclean ,                   (8)        of their predicted probabilities. For adversarial-clean align-
                                                                          ment, we pre-compute embedding statistics from the pub-
where α is a hyperparameter. Setting α = 0 aligns the
                                                                          lic dataset (ImageNet) using APT and standard PT, respec-
prompt with the clean distribution, which may reduce ad-
                                                                          tively. We then optimize the defensive prompts by minimiz-
versarial robustness, while setting α = 1 aligns the prompt
                                                                          ing a combined loss of multi-view entropy and adversarial-
with the robust distribution, which may impact performance
                                                                          clean alignment using the AdamW optimizer, with a learn-
on clean samples. Our final objective combines the multi-
                                                                          ing rate of 5 × 10−4 and an adversarial-clean scale factor of
view entropy loss with the adversarial-clean alignment to
                                                                          α = 0.5, on a single NVIDIA A100 GPU.
optimize the prompt during inference for a given test sam-
ple. This approach enhances adversarial robustness while
                                                                          4.2. Main Results
preserving accuracy on clean samples. Note that, to ensure
inference efficiency, TAPT performs only a single step of                 Zero-Shot Adversarial Robustness We compare our
prompt tuning for each inference.                                         TAPT method with existing APT methods across three


                                                                      5
                                                              ImageNet          Caltech101 DTD                           EuroSAT            Pets       Aircraft   Food101      Flowers      Cars         SUN397       UCF101       Avg.
                                         PGD 1.4                                21.3                    1.4              6.0                5.0        0.0        9.8          1.6          0.7          0.8          1.8          4.5
                                ViT-B/16 DI  6.1                                25.8                    8.7              0.3                11.1       0.5        9.5          3.0          3.4          4.9          4.7          7.1

 CLIP
                                         AA 0.0                                 0.0                     0.0              0.1                0.0        0.1        0.0          0.0          0.0          0.0          0.0          0.1
                   Vanilla
                                         PGD 1.3                                22.9                    5.0              0.0                2.6        0.0        3.6          1.5          0.2          1.1          1.7          3.6
                                ViT-B/32 DI  6.4                                37.2                    11.1             0.7                9.9        0.1        7.5          9.3          3.9          8.6          5.8          9.1
                                         AA 0.0                                 0.0                     0.0              0.1                0.0        0.1        0.0          0.1          0.1          0.0          0.1          0.1
                                         PGD 19.4         61.2                                          18.5         8.0          37.4         3.9          12.1         25.1         9.8          17.2         16.4         20.8
                                ViT-B/16 DI  29.2         67.5                                          23.5         12.3         47.1         5.0          21.8         30.2         18.1         27.1         22.0         27.6
                                         AA 14.8          55.9                                          15.2         2.3          31.3         2.3          8.3          18.0         5.8          12.4         12.7         16.3
                   APT-V
                                         PGD 18.9         63.8                                          20.2         0.6          36.1         2.7          15.0         23.1         9.4          19.1         18.4         20.7


 Visual Only
                                ViT-B/32 DI  26.9         68.6                                          22.0         5.1          46.0         4.8          23.7         26.5         13.9         26.4         24.6         26.2
                                         AA 8.3           44.9                                          12.7         0.1          16.4         0.5          5.5          9.0          2.7          7.5          7.8          10.5
                                         PGD 40.1 (20.7↑) 69.7 (8.5↑)                                   28.1 (9.6↑) 23.9 (15.9↑) 49.2 (11.8↑) 11.5 (7.6↑) 54.5 (42.4↑) 41.1 (16.0↑) 28.6 (18.8↑) 40.3 (23.1↑) 36.5 (20.1↑) 38.5 (17.7↑)
                                ViT-B/16 DI  46.7 (17.5↑) 75.8 (8.3↑)                                   33.2 (9.7↑) 33.0 (20.7↑) 56.7 (9.6↑) 12.7 (7.7↑) 61.5 (39.7↑) 46.2 (16.0↑) 35.7 (17.6↑) 45.5 (18.4↑) 40.1 (18.1↑) 44.3 (16.7↑)
                                         AA 49.2 (34.4↑) 75.7 (19.8↑)                                   36.6 (21.4↑) 36.9 (34.6↑) 57.1 (25.8↑) 19.4 (17.1↑) 68.8 (60.5↑) 52.9 (34.9↑) 44.0 (38.2↑) 48.9 (36.5↑) 48.1 (35.4↑) 48.9 (32.6↑)
                   TAPT-V
                                         PGD 42.2 (23.3↑) 79.4 (15.6↑)                                  32.2 (12.0↑) 24.4 (23.8↑) 62.4 (26.3↑) 12.1 (9.4↑) 53.2 (38.2↑) 47.1 (24.0↑) 33.5 (24.1↑) 44.1 (25.0↑) 44.5 (26.1↑) 43.2 (22.5↑)
                                ViT-B/32 DI  46.5 (19.6↑) 81.5 (12.9↑)                                  33.6 (11.6↑) 24.8 (19.7↑) 66.0 (20.0↑) 13.3 (8.5↑) 57.7 (34.0↑) 48.7 (22.2↑) 37.5 (23.6↑) 47.3 (20.9↑) 47.7 (23.1↑) 45.9 (19.7↑)
                                         AA 44.1 (35.8↑) 76.3 (31.4↑)                                   33.5 (20.8↑) 31.4 (31.3↑) 54.9 (38.5↑) 16.4 (15.9↑) 53.6 (48.1↑) 47.3 (38.3↑) 42.4 (39.7↑) 45.6 (38.1↑) 45.0 (37.2↑) 44.6 (34.1↑)
                                     PGD 23.9         61.7                                              18.7         9.7          41.4         3.2                14.1         23.4         12.2         17.7         15.5         22.0
                            ViT-B/16 DI  34.9         72.0                                              25.0         10.5         52.8         4.2                24.6         32.9         18.9         28.5         24.8         29.9
                                     AA 16.5          53.2                                              12.6         5.6          31.3         1.7                8.0          16.0         4.9          11.1         11.1         15.6
                   APT-VLJ
                                     PGD 21.4         64.3                                              14.7         10.4         37.7         2.0                16.5         21.0         8.9          17.5         18.0         21.1
                            ViT-B/32 DI  30.4         69.5                                              17.0         11.9         47.9         2.9                26.7         26.1         17.3         26.5         26.4         27.5

 V-L Joint
                                     AA 10.3          46.2                                              10.0         3.0          18.5         0.6                6.7          9.5          1.7          7.4          7.2          11.1
                                     PGD 50.2 (26.3↑) 81.0 (19.3↑)                                      29.5 (10.8↑) 13.5 (3.8↑) 68.7 (27.3↑) 5.6 (2.4↑)          41.7 (27.6↑) 39.3 (15.9↑) 28.1 (15.9↑) 39.8 (22.1↑) 41.5 (26.0↑) 39.9 (17.9↑)
                            ViT-B/16 DI  51.7 (16.8↑) 82.3 (10.3↑)                                      30.4 (5.4↑) 14.5 (4.0↑) 69.5 (16.7↑) 5.8 (1.6↑)           43.2 (18.6↑) 40.2 (7.3↑) 30.2 (11.3↑) 41.3 (12.8↑) 42.2 (17.4↑) 41.0 (11.1↑)
                                     AA 52.4 (35.9↑) 81.5 (28.3↑)                                       30.2 (17.6↑) 15.4 (9.8↑) 69.1 (37.8↑) 5.8 (4.1↑)          44.8 (36.8↑) 40.0 (24.0↑) 31.0 (26.1↑) 42.1 (31.0↑) 44.3 (33.2↑) 41.5 (25.9↑)
                   TAPT-VLJ
                                     PGD 52.1 (30.7↑) 80.6 (16.3↑)                                      27.1 (12.4↑) 13.4 (3.0↑) 72.0 (34.3↑) 7.6 (5.6↑)          52.5 (36.0↑) 39.9 (18.0↑) 32.3 (23.4↑) 44.6 (27.1↑) 45.3 (27.1↑) 42.4 (21.3↑)
                            ViT-B/32 DI  52.4 (22.0↑) 80.5 (11.0↑)                                      27.1 (10.1↑) 13.6 (1.7↑) 71.6 (23.7↑) 7.7 (4.8↑)          52.0 (25.3↑) 40.0 (13.9↑) 33.4 (16.1↑) 44.6 (18.1↑) 45.5 (19.1↑) 42.6 (15.1↑)
                                     AA 52.4 (42.1↑) 78.6 (32.4↑)                                       27.4 (17.4↑) 13.1 (10.1↑) 70.0 (51.5↑) 8.6 (8.0↑)         53.0 (46.3↑) 40.5 (31.0↑) 34.7 (33.0↑) 44.9 (37.5↑) 44.8 (37.6↑) 42.5 (31.4↑)
                                     PGD 24.3         65.3                                              18.9         10.0         43.6         3.1          14.3         23.6         10.5         18.2         17.4         22.7
                            ViT-B/16 DI  35.1         67.7                                              19.3         10.6         49.8         4.3          23.6         27.7         16.2         26.4         21.5         27.5
                                     AA 17.2          57.1                                              14.4         8.2          35.3         1.5          8.9          16.6         5.1          11.9         12.5         17.2
                   APT-VLI




 V-L Independent
                                     PGD 21.2         63.2                                              15.8         10.4         37.6         1.5          16.1         20.2         8.3          16.9         17.6         20.8
                            ViT-B/32 DI  28.9         69.5                                              20.3         10.9         47.0         2.8          24.8         24.3         14.8         23.8         23.6         26.4
                                     AA 9.7           45.6                                              10.6         6.8          17.3         0.4          6.0          8.7          2.0          6.8          6.9          11.0
                                     PGD 50.0 (25.7↑) 79.0 (13.7↑)                                      32.4 (13.5↑) 36.2 (26.2↑) 67.5 (51.4↑) 13.1 (10.0↑) 65.7 (51.4↑) 49.8 (26.2↑) 39.6 (29.1↑) 48.3 (30.1↑) 47.5 (30.1↑) 48.1 (25.4↑)
                            ViT-B/16 DI  53.8 (18.7↑) 80.5 (12.8↑)                                      32.3 (13.0↑) 39.6 (29.0↑) 69.4 (19.6↑) 13.3 (9.0↑) 70.6 (47.0↑) 51.1 (23.4↑) 42.5 (26.3↑) 50.3 (23.9↑) 48.2 (26.7↑) 50.1 (22.6↑)
                                     AA 55.1 (37.9↑) 80.3 (23.2↑)                                       35.7 (21.3↑) 44.4 (36.2↑) 70.2 (34.9↑) 16.0 (14.5↑) 76.2 (67.3↑) 55.1 (38.5↑) 50.5 (45.4↑) 53.6 (41.7↑) 54.5 (42.0↑) 53.8 (36.6↑)
                   TAPT-VLI
                                     PGD 48.2 (27.0↑) 82.1 (18.9↑)                                      30.7 (14.9↑) 31.6 (21.2↑) 68.1 (30.5↑) 4.5 (3.0↑)   64.7 (48.6↑) 44.6 (24.4↑) 41.3 (33.0↑) 47.6 (30.7↑) 49.1 (31.5↑) 46.6 (25.8↑)
                            ViT-B/32 DI  50.0 (21.1↑) 82.7 (13.2↑)                                      31.5 (11.2↑) 32.7 (21.8↑) 69.4 (22.4↑) 4.4 (1.6↑)   65.6 (40.8↑) 45.1 (20.8↑) 42.8 (28.0↑) 49.1 (25.3↑) 49.9 (26.3↑) 47.6 (21.2↑)
                                     AA 49.7 (40.0↑) 80.6 (35.0↑)                                       33.3 (22.7↑) 38.1 (31.3↑) 68.4 (51.1↑) 5.0 (4.6↑)   67.5 (61.5↑) 47.1 (38.4↑) 48.1 (46.1↑) 50.3 (43.5↑) 50.4 (43.5↑) 49.0 (38.0↑)


Table 1. Zero-shot adversarial robustness (%) of different defense methods from ImageNet to downstream datasets, evaluated against
PGD [25], DI [53], and AutoAttack (AA) [8] under perturbation budget ϵ = 1/255. The baseline APT methods (APT-V, APT-VLJ, and
APT-VLI) were tuned on ImageNet under a 16-shot setting and then assessed on the other 10 datasets. The green upward arrows (↑)
highlight the performance improvement of our TAPT over the baselines.


                             Source                                                         Target                                                     tack attacks, with the ‘Vanilla’ showing results without any

                                           Caltech101
                                                                                                                                                       defense. It is clear that both white-box and black-box at-
                               ImageNet                 DTD    EuroSAT           Aircraft     Food101   Flowers          SUN397   UCF101
                                                                                                                                                       tacks can drastically reduce accuracy (nearly 0%) in the
                                                                         Pets                                     Cars                     Avg.
                                                                                                                                                       absence of defenses, using only imperceptible noise with
  APT-V                       60.6        89.7 36.9 26.7 84.8 16.8 63.2 55.7 52.1 58.2 54.4 54.5
  TAPT-V                      66.9        92.8 44.6 41.0 88.5 23.8 85.7 67.4 65.7 62.9 65.5 64.1
                                                                                                                                                       ϵ = 1/255. AutoAttack emerges as the most effective at-
  APT-VLJ                     64.0        88.5 34.8 16.0 81.1 8.1 62.6 53.3 47.0 53.0 52.8 51.0                                                        tack, achieving a nearly 100% attack success rate on aver-
  TAPT-VLJ                    66.5        88.5 36.7 16.7 80.6 7.4 64.6 50.4 48.7 54.7 55.4 51.8                                                        age across all datasets. For defense, our TAPT method con-
  APT-VLI                     63.8        88.6 34.1 17.2 80.7 11.6 61.1 51.8 45.9 53.5 52.7 51.0                                                       sistently achieves the best average performance under Au-
  TAPT-VLI                    65.8        90.4 39.2 48.3 82.5 16.3 85.8 60.3 62.5 60.0 64.4 61.4
                                                                                                                                                       toAttack across all prompt designs (visual-only, V-L joint,
Table 2. Zero-shot clean accuracy (%) of different defense meth-                                                                                       and V-L independent) and ViT architectures (ViT-B/16 and
ods from ImageNet to downstream datasets, including APT base-                                                                                          ViT-B/32). With ViT-B/16, TAPT enhances robustness by
lines (APT-V, APT-VLJ, and APT-VLI) and our TAPT. The back-                                                                                            32.6% (visual-only), 25.9% (V-L joint), and 36.6% (V-L
bone is ViT-B/16. The best results are boldfaced.                                                                                                      independent), respectively. Similar improvements are ob-
                                                                                                                                                       served with ViT-B/32, where TAPT yields gains of 34.1%,
                                                                                                                                                       31.4%, and 38.0% for the respective prompt designs. TAPT
prompt designs: visual-only (V), V-L joint (VLJ), and V-                                                                                               also demonstrates superior performance against both PGD-
L independent (VLI). Table 1 presents the zero-shot adver-                                                                                             100 and DI attacks.
sarial robustness results under PGD-100, DI, and AutoAt-                                                                                                   Furthermore, TAPT with the V-L independent prompt


                                                                                                                                                   6
                                     (a) Average                                                (b) ImageNet                                                  (c) Caltech101                                                  (d) DTD
                      55
                                                                                                                                                   85                                                         35
                      50
                                                                                 50




Robust Accuracy (%)                                        Robust Accuracy (%)                                               Robust Accuracy (%)                                        Robust Accuracy (%)
                      45                                                                                                                           80
                                                                                                                                                                                                              30
                      40                                                         40                                                                75
                      35                                                                                                                                                                                      25
                                                                                                                                                   70
                      30                                                         30
                                                                                                                                                                                                              20
                      25                                                                                                                           65
                      20                                                         20                                                                                                                           15
                                                                                                                                                   60
                           0     1        2           4                               0     1        2           4                                      0     1        2            4                              0     1        2           4
                               Number of TAPT Steps                                       Number of TAPT Steps                                              Number of TAPT Steps                                       Number of TAPT Steps
                                     (e) EuroSAT                                                  (f) Pets                                                        (g) Aircraft                                               (h) Food101
                      40
                                                                                                                                           15.0                                                               70
                                                                                 70




Robust Accuracy (%)                                        Robust Accuracy (%)                                       Robust Accuracy (%)                                                Robust Accuracy (%)
                      30                                                                                                                   12.5                                                               60
                                                                                 60                                                        10.0                                                               50
                      20
                                                                                                                                             7.5                                                              40
                                                                                 50
                      10                                                                                                                     5.0                                                              30
                                                                                 40                                                                                                                           20
                                                                                                                                             2.5
                       0                                                                                                                                                                                      10
                           0     1        2           4                               0     1        2           4                                      0     1        2            4                              0     1        2           4
                               Number of TAPT Steps                                       Number of TAPT Steps                                              Number of TAPT Steps                                       Number of TAPT Steps
                                      (i) Flowers                                                 (j) Cars                                                        (k) SUN397                                                 (l) UCF101
                      55                                                         50
                      50                                                                                                                           50                                                         50




Robust Accuracy (%)                                        Robust Accuracy (%)                                               Robust Accuracy (%)                                        Robust Accuracy (%)
                      45                                                         40
                      40                                                                                                                           40                                                         40
                                                                                 30
                      35
                                                                                                                                                   30                                                         30
                      30                                                         20
                      25                                                                                                                                                                                      20
                                                                                 10                                                                20
                      20
                           0     1        2           4                               0     1        2           4                                      0     1        2            4                              0     1        2           4
                               Number of TAPT Steps                      Number of TAPT Steps                                                               Number of TAPT Steps                                       Number of TAPT Steps
                                                          Visual Prompt ViT-B/16       V-L Joint Prompt ViT-B/16                                                V-L Independent Prompt ViT-B/16
                                                          Visual Prompt ViT-B/32       V-L Joint Prompt ViT-B/32                                                V-L Independent Prompt ViT-B/32

 Figure 4. Adversarial robustness (%) of our TAPT method under different test-time adaptation steps (i.e., {0, 1, 2, 4}). The results are
 reported against the PGD-100 attack on ViT-B/16 and ViT-B/32 architectures.

 design achieves the highest average zero-shot adversar-                                                                                           bust statistics were computed based on ImageNet. The
 ial robustness, surpassing both visual-only and V-L joint                                                                                         zero-shot clean accuracy should be compared on all 10
 prompt designs. To summarize, our experiments reveal that:                                                                                        datasets. As shown, TAPT consistently achieves superior
 (1) incorporating textual prompts generally improves zero-                                                                                        clean performance across all 11 datasets, demonstrating
 shot adversarial robustness across various datasets; (2) the                                                                                      significantly stronger generalization capabilities. While a
 V-L joint prompt enhances robustness on the source domain                                                                                         slight performance decrease is observed with TAPT’s V-L
 (ImageNet) more effectively than other prompt designs; and                                                                                        joint prompt design on Pets, Aircraft, and Flowers, it re-
 (3) the V-L independent prompt design is more effective                                                                                           mains competitive. Overall, TAPT outperforms APT across
 in enhancing zero-shot adversarial robustness than the V-L                                                                                        all three prompt designs (visual-only, V-L joint, and V-L in-
 joint prompt design, likely due to the challenges in optimiz-                                                                                     dependent), with average improvements of 9.6%, 0.8%, and
 ing the interplay between visual and textual prompts.                                                                                             10.4%, respectively. These results underscore TAPT’s abil-
                                                                                                                                                   ity to improve adversarial robustness without compromising
 Zero-Shot Clean Accuracy Table 2 presents the zero-shot                                                                                           too much clean accuracy.
 clean accuracy of different defense methods from ImageNet
 to downstream datasets, comparing APT baselines (APT-                                                                                             4.3. Ablation Studies
 V, APT-VLJ, and APT-VLI) with our TAPT method on
 the ViT-B/16 architecture. Note that the APT baselines                                                                                            Number of TAPT Steps We first examine the effect of the
 were trained on ImageNet and subsequently tested on 10                                                                                            number of test-time adaptation steps on the robust accuracy
 downstream datasets, while for our TAPT, only the ro-                                                                                             of TAPT. Figure 4 shows TAPT’s robustness performance


                                                                                                                         7
      with varying steps (0, 1, 2, and 4) under the PGD-100 at-                                                                                    bust accuracy, suggesting that accumulating information
      tack. Notably, step = 0 represents the baseline perfor-                                                                                      across multiple inputs can be beneficial. However, this con-
      mance without TAPT, reflecting the adversarial robustness                                                                                    tinuous strategy introduces a vulnerability to potential poi-
      achieved solely through APT. A clear trend emerges, show-                                                                                    soning attacks. To mitigate this risk and ensure reliable test-
      ing that robust accuracy increases with additional TAPT                                                                                      time defense, we recommend the per-sample reset strategy
      steps across most datasets and prompt designs. While ro-                                                                                     (“reset=1”), which prioritizes robustness against potential
      bustness gains generally stabilize after a few TAPT steps,                                                                                   attacks over marginal gains from continued TAPT.
      the improvement from 0 to 1 step is consistently substantial,
      highlighting the effectiveness of even a single adaptation                                                                                        Reset Interval   ImageNet    10 Zero-Shot Datasets Avg.
      step. However, datasets vary in sensitivity to the number                                                                                         reset=1            49.92                47.85
      of TAPT steps; for example, performance stabilizes quickly                                                                                        reset=2            50.20                47.91
      on datasets like EuroSAT, whereas for datasets such as Im-                                                                                        reset=4            50.69                47.91
                                                                                                                                                        reset=8            51.14                47.53
      ageNet and DTD, further improvements are observed with                                                                                            reset=16           51.49                46.39
      additional steps. The figure also illustrates that TAPT’s ben-                                                                                    reset=32           51.62                44.50
      efits are consistent across different prompt designs. We fur-                                                                                     reset=all           0.48                 3.79
      ther analyzed the per-sample time overhead of TAPT, find-
      ing that the additional inference time per image is 0.095s                                                                                   Table 3. Zero-shot adversarial robustness (%) of TAPT with vary-
      (visual-only), 0.166s (V-L joint), and 0.165s (V-L indepen-                                                                                  ing reset intervals on ImageNet and 10 other zero-shot datasets.
      dent), respectively. This demonstrates that TAPT not only                                                                                    “reset = N ” means the prompt is reset after every N test samples.
      consistently enhances robust accuracy but also maintains a
      relatively low time cost.
                                                                                                                                                   5. Limitation
      Different Perturbation Budgets We further assess TAPT’s
      robustness under varying attack strengths, defined by the                                                                                    As a test-time defense method, TAPT has certain limita-
      perturbation budget ϵ. Figure 5 displays zero-shot adver-                                                                                    tions that warrant further research. Our method primar-
      sarial robustness across 11 datasets with ϵ values of 1/255,                                                                                 ily addresses attacks in the image modality by aligning ad-
      2/255, and 4/255, and TAPT steps of 1, 2, and 4. As ex-                                                                                      versarial image embeddings with pre-computed public data
      pected, robust accuracy decreases as ϵ increases, indicating                                                                                 statistics on a public dataset. Future work could explore
      stronger attacks. Nonetheless, TAPT consistently enhances                                                                                    additional modality alignment and acceleration techniques
      robust accuracy across all datasets and ϵ values.                                                                                            to facilitate TAPT’s deployment in industrial applications.
                                                                                                                                                   Moreover, our current focus is limited to image recogni-
                            80                                                      = 1/255, step1        = 2/255, step1      = 4/255, step1       tion tasks. Extending TAPT to a broader range of tasks,




Zero-Shot Robust Accuracy (%)
                                                                                    = 1/255, step2        = 2/255, step2      = 4/255, step2
                                                                                    = 1/255, step4        = 2/255, step4      = 4/255, step4
                            70                                                                                                                     such as visual reasoning and visual question answering in
                            60
                                                                                                                                                   advanced models like GPT-4V [2] and Gemini [44], repre-
                            50
                            40
                                                                                                                                                   sents a promising direction for future research.
                            30
                            20                                                                                                                     6. Conclusion
                            10
                                0
                                                                                                                                                   In this paper, we introduced a novel test-time defense
                                      et             D             Pe                    01          rs                97
                                                                                                                                                   method, Test-Time Adversarial Prompt Tuning (TAPT), to
                                             10  1        roS         ts      ft                          rdC                 01
                                    agen    ch       DT       AT           Aircra   Fo          Flowe        ars   SU       UC
                                           lte            Eu                           od 1             nfo          N3       F1
                                Im     Ca
                                                                             Datasets
                                                                                                     Sta                                           enhance the inference robustness of pre-trained VLMs, such
                                                                                                                                                   as CLIP. TAPT tunes defensive bimodal (textual and vi-
      Figure 5. Zero-shot adversarial robustness (y-axis) ofs TAPT un-                                                                             sual) prompts for each test sample by leveraging multi-view
      der varying perturbation budgets ϵ (1/255, 2/255, and 4/255) and                                                                             entropy minimization and adversarial-clean alignment, ef-
      TAPT steps (1, 2, and 4).                                                                                                                    fectively safeguarding CLIP’s zero-shot inference. It also
                                                                                                                                                   utilizes pre-computed statistics from a public dataset (Ima-
      TAPT Reset Intervals Our TAPT method resets the prompt                                                                                       geNet) to defend a wide range of downstream tasks. Com-
      to its initial state before processing each test sample, ensur-                                                                              prehensive evaluation across 11 benchmark datasets demon-
      ing that each test sample is handled independently. How-                                                                                     strates that TAPT effectively enhances zero-shot adversar-
      ever, we also explored alternative strategies with varied re-                                                                                ial robustness against both white-box and black-box at-
      set intervals. As shown in Table 3, we experimented with                                                                                     tacks, while largely preserving clean accuracy. Compared
      reset intervals ranging from 1 to 32, as well as a strat-                                                                                    to training-time adversarial prompt tuning (APT) methods,
      egy with no reset during the entire inference process (i.e.,                                                                                 TAPT offers several advantages: (1) it is unsupervised, (2)
      “reset=all”). Our findings indicate that continuous prompt                                                                                   enables sample-wise prompt adaptation, (3) delivers supe-
      adaptation without frequent resets can further improve ro-                                                                                   rior zero-shot adversarial robustness and clean accuracy, (4)


                                                                                                                                               8
is independent of downstream tasks, and (5) is lightweight.                    benchmark for land use and land cover classification. IEEE
Future research could explore the scalability of test-time de-                 J-STARS, 2019. 5
fense across different modalities.                                        [15] Zhi Huang, Federico Bianchi, Mert Yuksekgonul, Thomas J
                                                                               Montine, and James Zou. A visual–language foundation
References                                                                     model for pathology image analysis using medical twitter.
                                                                               Nature Medicine, 2023. 1
 [1] Jameel Abdul Samadh, Mohammad Hanan Gani, Noor Hus-                  [16] Chao Jia, Yinfei Yang, Ye Xia, Yi-Ting Chen, Zarana Parekh,
     sein, Muhammad Uzair Khattak, Muhammad Muzammal                           Hieu Pham, Quoc Le, Yun-Hsuan Sung, Zhen Li, and Tom
     Naseer, Fahad Shahbaz Khan, and Salman H Khan. Align                      Duerig. Scaling up visual and vision-language representation
     your prompts: Test-time prompting with distribution align-                learning with noisy text supervision. In ICML, 2021. 1
     ment for zero-shot generalization. In NeurIPS, 2024. 3, 4            [17] Apoorv Khandelwal, Luca Weihs, Roozbeh Mottaghi, and
 [2] Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ah-                     Aniruddha Kembhavi. Simple but effective: Clip embed-
     mad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida,                  dings for embodied ai. In CVPR, 2022. 1
     Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al.              [18] Muhammad Uzair Khattak, Hanoona Rasheed, Muhammad
     Gpt-4 technical report. preprint arXiv:2303.08774, 2023. 8                Maaz, Salman Khan, and Fahad Shahbaz Khan. Maple:
 [3] Michael Ahn, Anthony Brohan, Noah Brown, Yevgen Cheb-                     Multi-modal prompt learning. In CVPR, 2023. 1
     otar, Omar Cortes, Byron David, Chelsea Finn, Chuyuan Fu,            [19] Eungyeup Kim, Mingjie Sun, Christina Baek, Aditi Raghu-
     Keerthana Gopalakrishnan, Karol Hausman, et al. Do as i                   nathan, and J Zico Kolter. Test-time adaptation induces
     can, not as i say: Grounding language in robotic affordances.             stronger accuracy and agreement-on-the-line. In NeurIPS,
     preprint arXiv:2204.01691, 2022. 1                                        2024. 3
 [4] Christina Baek, Yiding Jiang, Aditi Raghunathan, and J Zico          [20] Hoki Kim. Torchattacks: A pytorch repository for adversar-
     Kolter. Agreement-on-the-line: Predicting the performance                 ial attacks. preprint arXiv:2010.01950, 2020. 5
     of neural networks under distribution shift. In NeurIPS,             [21] Jonathan Krause, Michael Stark, Jia Deng, and Li Fei-Fei.
     2022. 3                                                                   3d object representations for fine-grained categorization. In
 [5] Hyojin Bahng, Ali Jahanian, Swami Sankaranarayanan, and                   ICCV Workshops, 2013. 5
     Phillip Isola. Exploring visual prompts for adapting large-          [22] Lin Li, Haoyan Guan, Jianing Qiu, and Michael Spratling.
     scale models. preprint arXiv:2203.17274, 2022. 5                          One prompt word is enough to boost adversarial robustness
 [6] Lukas Bossard, Matthieu Guillaumin, and Luc Van Gool.                     for pre-trained vision-language models. In CVPR, 2024. 1,
     Food-101–mining discriminative components with random                     2, 5
     forests. In ECCV, 2014. 5                                            [23] Dong Lu, Zhiqiang Wang, Teng Wang, Weili Guan,
 [7] Mircea Cimpoi, Subhransu Maji, Iasonas Kokkinos, Sammy                    Hongchang Gao, and Feng Zheng. Set-level guidance at-
     Mohamed, and Andrea Vedaldi. Describing textures in the                   tack: Boosting adversarial transferability of vision-language
     wild. In CVPR, 2014. 5                                                    pre-training models. In ICCV, 2023. 2
 [8] Francesco Croce and Matthias Hein. Reliable evalua-                  [24] Xingjun Ma, Linxi Jiang, Hanxun Huang, Zejia Weng, James
     tion of adversarial robustness with an ensemble of diverse                Bailey, and Yu-Gang Jiang. Imbalanced gradients: a sub-
     parameter-free attacks. In ICML, 2020. 1, 2, 5, 6                         tle cause of overestimated adversarial robustness. Machine
 [9] Yinpeng Dong, Fangzhou Liao, Tianyu Pang, Hang Su, Jun                    Learning, 2024. 2
     Zhu, Xiaolin Hu, and Jianguo Li. Boosting adversarial at-            [25] Aleksander Madry, Aleksandar Makelov, Ludwig Schmidt,
     tacks with momentum. In CVPR, 2018. 1                                     Dimitris Tsipras, and Adrian Vladu. Towards deep learning
[10] Hao Fang, Jiawei Kong, Wenbo Yu, Bin Chen, Jiawei Li,                     models resistant to adversarial attacks. ICLR, 2018. 1, 2, 5,
     Shutao Xia, and Ke Xu. One perturbation is enough: On                     6
     generating universal adversarial perturbations against vision-       [26] Subhransu Maji, Esa Rahtu, Juho Kannala, Matthew
     language pre-training models. preprint arXiv:2406.05491,                  Blaschko, and Andrea Vedaldi. Fine-grained visual classi-
     2024. 2                                                                   fication of aircraft. preprint arXiv:1306.5151, 2013. 5
[11] Li Fei-Fei, Rob Fergus, and Pietro Perona. Learning gener-           [27] Chengzhi Mao, Scott Geng, Junfeng Yang, Xin Wang, and
     ative visual models from few training examples: An incre-                 Carl Vondrick. Understanding zero-shot adversarial robust-
     mental bayesian approach tested on 101 object categories. In              ness for large-scale models. In ICLR, 2023. 2, 5
     CVPR Workshops, 2004. 5                                              [28] Zachary Nado, Shreyas Padhy, D Sculley, Alexander
[12] Zhe Gan, Yen-Chun Chen, Linjie Li, Chen Zhu, Yu Cheng,                    D’Amour, Balaji Lakshminarayanan, and Jasper Snoek.
     and Jingjing Liu. Large-scale adversarial training for vision-            Evaluating prediction-time batch normalization for robust-
     and-language representation learning. In NeurIPS, 2020. 1                 ness under covariate shift. preprint arXiv:2006.10963, 2020.
[13] Bangyan He, Xiaojun Jia, Siyuan Liang, Tianrui Lou, Yang                  3
     Liu, and Xiaochun Cao. Sa-attack: Improving adversar-                [29] Maria-Elena Nilsback and Andrew Zisserman. Automated
     ial transferability of vision-language pre-training models via            flower classification over a large number of classes. In
     self-augmentation. preprint arXiv:2312.04913, 2023. 2                     ICVGIP, 2008. 5
[14] Patrick Helber, Benjamin Bischke, Andreas Dengel, and                [30] Shuaicheng Niu, Jiaxiang Wu, Yifan Zhang, Yaofo Chen,
     Damian Borth. Eurosat: A novel dataset and deep learning                  Shijian Zheng, Peilin Zhao, and Mingkui Tan. Efficient test-


                                                                      9
     time model adaptation without forgetting. In ICML, 2022.             [47] Qin Wang, Olga Fink, Luc Van Gool, and Dengxin Dai. Con-
     3                                                                         tinual test-time domain adaptation. In CVPR, 2022. 3
[31] Omkar M Parkhi, Andrea Vedaldi, Andrew Zisserman, and                [48] Sibo Wang, Jie Zhang, Zheng Yuan, and Shiguang Shan. Pre-
     CV Jawahar. Cats and dogs. In CVPR, 2012. 5                               trained model guided fine-tuning for zero-shot adversarial
[32] Anibal Pedraza, Oscar Deniz, and Gloria Bueno. On the rela-               robustness. In CVPR, 2024. 2
     tionship between generalization and robustness to adversar-          [49] Xin Wang, Kai Chen, Xingjun Ma, Zhineng Chen, Jingjing
     ial examples. Symmetry, 2021. 1                                           Chen, and Yu-Gang Jiang. AdvQDet: Detecting query-based
[33] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya                        adversarial attacks with adversarial contrastive prompt tun-
     Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry,                     ing. In ACM MM, 2024. 2
     Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learn-             [50] Zifeng Wang, Zhenbang Wu, Dinesh Agarwal, and Jimeng
     ing transferable visual models from natural language super-               Sun. Medclip: Contrastive learning from unpaired medical
     vision. In ICML, 2021. 1, 3                                               images and text. In EMNLP, 2022. 1
[34] Olga Russakovsky, Jia Deng, Hao Su, Jonathan Krause, San-            [51] Zeyu Wang, Xianhang Li, Hongru Zhu, and Cihang Xie. Re-
     jeev Satheesh, Sean Ma, Zhiheng Huang, Andrej Karpathy,                   visiting adversarial training at scale. In CVPR, 2024. 1, 2
     Aditya Khosla, Michael Bernstein, et al. Imagenet large              [52] Jianxiong Xiao, James Hays, Krista A Ehinger, Aude Oliva,
     scale visual recognition challenge. IJCV, 2015. 5                         and Antonio Torralba. Sun database: Large-scale scene
[35] Christian Schlarmann, Naman Deep Singh, Francesco                         recognition from abbey to zoo. In CVPR, 2010. 5
     Croce, and Matthias Hein. Robust CLIP: Unsupervised ad-              [53] Cihang Xie, Zhishuai Zhang, Yuyin Zhou, Song Bai, Jianyu
     versarial fine-tuning of vision embeddings for robust large               Wang, Zhou Ren, and Alan L Yuille. Improving transferabil-
     vision-language models. In ICML, 2024. 2                                  ity of adversarial examples with input diversity. In CVPR,
[36] Steffen Schneider, Evgenia Rusak, Luisa Eck, Oliver Bring-                2019. 2, 5, 6
     mann, Wieland Brendel, and Matthias Bethge. Improving
                                                                          [54] Ziyi Yin, Muchao Ye, Tianrong Zhang, Tianyu Du, Jinguo
     robustness against common corruptions by covariate shift
                                                                               Zhu, Han Liu, Jinghui Chen, Ting Wang, and Fenglong Ma.
     adaptation. NeurIPS, 2020. 3
                                                                               Vlattack: Multimodal adversarial attacks on vision-language
[37] Or Sharir, Barak Peleg, and Yoav Shoham. The cost                         tasks via pre-trained models. In NeurIPS, 2023. 2
     of training nlp models: A concise overview. preprint
                                                                          [55] Gengyuan Zhang, Jisen Ren, Jindong Gu, and Volker Tresp.
     arXiv:2004.08900, 2020. 1
                                                                               Multi-event video-text retrieval. In ICCV, 2023. 1
[38] Mohit Shridhar, Lucas Manuelli, and Dieter Fox. Cliport:
                                                                          [56] Hongyang Zhang, Yaodong Yu, Jiantao Jiao, Eric Xing, Lau-
     What and where pathways for robotic manipulation. In
                                                                               rent El Ghaoui, and Michael Jordan. Theoretically principled
     CoRL, 2022. 1
                                                                               trade-off between robustness and accuracy. In ICML, 2019.
[39] Manli Shu, Weili Nie, De-An Huang, Zhiding Yu, Tom
                                                                               1
     Goldstein, Anima Anandkumar, and Chaowei Xiao. Test-
     time prompt tuning for zero-shot generalization in vision-           [57] Jiaming Zhang, Qi Yi, and Jitao Sang. Towards adversarial
     language models. In NeurIPS, 2022. 3, 4                                   attack on vision-language pre-training models. In ACM MM,
                                                                               2022. 1, 2
[40] K Soomro. Ucf101: A dataset of 101 human actions classes
     from videos in the wild. preprint arXiv:1212.0402, 2012. 5           [58] Jiaming Zhang, Xingjun Ma, Xin Wang, Lingyu Qiu, Jiaqi
[41] David Stutz, Matthias Hein, and Bernt Schiele. Disentan-                  Wang, Yu-Gang Jiang, and Jitao Sang. Adversarial prompt
     gling adversarial robustness and generalization. In CVPR,                 tuning for vision-language models. In ECCV, 2024. 1, 2, 5
     2019. 1                                                              [59] Marvin Zhang, Sergey Levine, and Chelsea Finn. Memo:
[42] Dong Su, Huan Zhang, Hongge Chen, Jinfeng Yi, Pin-Yu                      Test time robustness via adaptation and augmentation. In
     Chen, and Yupeng Gao. Is robustness the cost of accuracy?–                NeurIPS, 2022. 3
     a comprehensive study on the robustness of 18 deep image             [60] Peng-Fei Zhang, Zi Huang, and Guangdong Bai. Univer-
     classification models. In ECCV, 2018. 1                                   sal adversarial perturbations for vision-language pre-trained
[43] Christian Szegedy, Wojciech Zaremba, Ilya Sutskever, Joan                 models. In ACM SIGIR, pages 862–871, 2024. 2
     Bruna, Dumitru Erhan, Ian Goodfellow, and Rob Fergus. In-            [61] Yunqing Zhao, Tianyu Pang, Chao Du, Xiao Yang, Chongx-
     triguing properties of neural networks. In ICLR, 2013. 1                  uan Li, Ngai-Man Man Cheung, and Min Lin. On evaluating
[44] Gemini Team, Rohan Anil, Sebastian Borgeaud, Jean-                        adversarial robustness of large vision-language models. In
     Baptiste Alayrac, Jiahui Yu, Radu Soricut, Johan Schalk-                  NeurIPS, 2024. 1, 2
     wyk, Andrew M Dai, Anja Hauth, Katie Millican, et al.                [62] Kaiyang Zhou, Jingkang Yang, Chen Change Loy, and Zi-
     Gemini: a family of highly capable multimodal models.                     wei Liu. Conditional prompt learning for vision-language
     arXiv:2312.11805, 2023. 8                                                 models. In CVPR, 2022. 1
[45] Dequan Wang, Evan Shelhamer, Shaoteng Liu, Bruno Ol-                 [63] Kaiyang Zhou, Jingkang Yang, Chen Change Loy, and Ziwei
     shausen, and Trevor Darrell. Tent: Fully test-time adaptation             Liu. Learning to prompt for vision-language models. IJCV,
     by entropy minimization. In ICLR, 2021. 3                                 2022. 1
[46] Haodi Wang, Kai Dong, Zhilei Zhu, Haotong Qin, Aishan                [64] Wanqi Zhou, Shuanghao Bai, Qibin Zhao, and Badong
     Liu, Xiaolin Fang, Jiakai Wang, and Xianglong Liu. Trans-                 Chen.      Revisiting the adversarial robustness of vision
     ferable multimodal attack on vision-language pre-training                 language models: a multimodal perspective. preprint
     models. In IEEE S&P, 2024. 2                                              arXiv:2404.19287, 2024. 2


                                                                     10
[65] Yiwei Zhou, Xiaobo Xia, Zhiwei Lin, Bo Han, and
     Tongliang Liu. Few-shot adversarial prompt learning on
     vision-language models. In NeurIPS, 2024. 1, 2
[66] Ziqi Zhou, Shengshan Hu, Minghui Li, Hangtao Zhang,
     Yechao Zhang, and Hai Jin. Advclip: Downstream-agnostic
     adversarial examples in multimodal contrastive learning. In
     ACM MM, 2023. 2




                                                                   11
