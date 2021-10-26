import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots(1, 1)
#set axis limits of plot (x=0 to 20, y=0 to 20)
plt.axis([-1, 16, 43, 58.5])
plt.xticks([0, 2, 4, 6, 8, 10, 12, 14, 16], [0, 1, 2, 3, 4, 5, 6, 7, 8])
plt.yticks([45, 47, 49, 51, 53, 55, 57, 59])

# plt.xticks(np.arange(40, 60, 10.0))

plt.gca().set_aspect('equal', adjustable='box')
#define circles
# c1=plt.Circle((5, 5), radius=1)
# c2=plt.Circle((10, 10), radius=2)
# c3=plt.Circle((15, 13), radius=3)
Transformer = 54.02
Local_Attention = 44.22
Sparse_Trans = 49.15
Longformer = 52.6
Linformer = 51.14
Reformer = 49.99
Sinkhorn_Trans = 50.89
Synthesizer = 52.43
BigBird = 53.94
Linear_Trans = 49.92
Performer = 50.81
luna_16 = 56.95
luna_128 = 57.42
luna_256 = 57.49

Transformer_speed = 1
Local_Attention_speed = 5.285714286
Linformer_speed = 5.5
Reformer_speed = 0.7857142857
Sinkhorn_Trans_speed = 3.785714286
Synthesizer_speed = 1.357142857
BigBird_speed = 1.071428571
Linear_Trans_speed = 5.571428571
Performer_speed = 5.714285714
luna_16_speed = 5.5
luna_128_speed = 5.1
luna_256_speed = 4.9

Transformer_memory = 1.2
Local_Attention_memory = 0.1445147679*1.2
Linformer_memory = 0.1044303797*1.2
Reformer_memory = 0.2405063291*1.2
Sinkhorn_Trans_memory = 0.1561181435*1.2
Synthesizer_memory = 0.7373417722*1.2
BigBird_memory = 0.3037974684*1.2
Linear_Trans_memory = 0.108649789*1.2
Performer_memory = 0.111814346*1.2
luna_16_memory = 0.1*1.2
luna_128_memory = 0.14*1.2
luna_256_memory = 0.16*1.2

c1=plt.Circle((5, 5), radius=1)

#add circles to plot
plt.gca().add_artist(plt.Circle((Transformer_speed*2, Transformer), radius=Transformer_memory, alpha=0.5, linewidth=1.5, ec='black', color='tab:blue'))
plt.text(Transformer_speed*2, Transformer+1.5, 'Transformer', fontsize=12)
plt.gca().add_artist(plt.Circle((Local_Attention_speed*2, Local_Attention), radius=Local_Attention_memory, alpha=0.5, linewidth=1.5, ec='black', color='cyan'))
plt.text(Local_Attention_speed*2, Local_Attention+Local_Attention_memory, 'Local Attention', fontsize=12)
plt.gca().add_artist(plt.Circle((Linformer_speed*2, Linformer), radius=Linformer_memory, alpha=0.5, linewidth=1.5, ec='black', color='tomato'))
plt.text(Linformer_speed*2-6*Linformer_memory, Linformer+2*Linformer_memory, 'Linformer', fontsize=12)
plt.gca().add_artist(plt.Circle((Reformer_speed*2, Reformer), radius=Reformer_memory, alpha=0.5, linewidth=1.5, ec='black', color='tab:olive'))
plt.text(Reformer_speed*2, Reformer+Reformer_memory, 'Reformer', fontsize=12)
plt.gca().add_artist(plt.Circle((Sinkhorn_Trans_speed*2, Sinkhorn_Trans), radius=Sinkhorn_Trans_memory, alpha=0.5, linewidth=1.5, ec='black', color='lime'))
plt.text(Sinkhorn_Trans_speed*2, Sinkhorn_Trans+Sinkhorn_Trans_memory, 'Sinkhorn', fontsize=12)
plt.gca().add_artist(plt.Circle((Synthesizer_speed*2, Synthesizer), radius=Synthesizer_memory, alpha=0.5, linewidth=1.5, ec='black', color='tab:orange'))
plt.text(Synthesizer_speed*2, Synthesizer, 'Synthesizer', fontsize=12)
plt.gca().add_artist(plt.Circle((BigBird_speed*2, BigBird), radius=BigBird_memory, alpha=0.5, linewidth=1.5, ec='black', color='tab:pink'))
plt.text(BigBird_speed*2+BigBird_memory, BigBird, 'BigBird', fontsize=12)
plt.gca().add_artist(plt.Circle((Linear_Trans_speed*2, Linear_Trans), radius=Linear_Trans_memory, alpha=0.5, linewidth=1.5, ec='black', color='chocolate'))
plt.text(Linear_Trans_speed*2-10*Linear_Trans_memory, Linear_Trans-5*Linear_Trans_memory, 'Linear Transformer', fontsize=12)
plt.gca().add_artist(plt.Circle((Performer_speed*2, Performer), radius=Performer_memory, alpha=0.5, linewidth=1.5, ec='black', color='tab:green'))
plt.text(Performer_speed*2+2*Performer_memory, Performer, 'Performer', fontsize=12)
plt.gca().add_artist(plt.Circle((luna_16_speed*2, luna_16), radius=luna_16_memory, alpha=0.8, linewidth=1.5, ec='black', color='red'))
plt.text(luna_16_speed*2 + 2*luna_16_memory, luna_16, 'Luna-16', fontsize=12)
plt.gca().add_artist(plt.Circle((luna_128_speed*2, luna_128), radius=luna_128_memory, alpha=0.8, linewidth=1.5, ec='black', color='red'))
plt.text(luna_128_speed*2+ 2*luna_128_memory, luna_128, 'Luna-128', fontsize=12)
plt.gca().add_artist(plt.Circle((luna_256_speed*2, luna_256), radius=luna_256_memory, alpha=0.8, linewidth=1.5, ec='black', color='red'))
plt.text(luna_256_speed*2-6*luna_256_memory, luna_256+2*luna_256_memory, 'Luna-256', fontsize=12)
plt.gca().add_artist(plt.Rectangle((7.6, 56.5), 7.5, 2.3, alpha=1, ec='red', facecolor='none', linestyle="dotted"))
# plt.text(7, 58, 'Our Proposed Work', fontsize=14, weight='bold')
plt.grid(color='gainsboro', ls='--')
plt.xlabel("Relative Speed Comparision", fontsize=14)
plt.ylabel("Avg. LRA Score (w/o Retrieval)", fontsize=14)
ax.spines["top"].set_color('gainsboro')
ax.spines["right"].set_color('gainsboro')
ax.spines["bottom"].set_color('gainsboro')
ax.spines["left"].set_color('gainsboro')
ax.xaxis.set_ticks_position('none')
ax.yaxis.set_ticks_position('none')  
fig.set_size_inches(10, 6)
fig.savefig("overall_tradeoff.pdf", bbox_inches='tight')

# plt.savefig('overall_tradeoff.pdf', bbox_inches='tight', pad_inches=0.1, edgecolor="gainsboro")
# plt.show()
  


