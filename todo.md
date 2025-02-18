1. Basic two layer GAT: 26.88%
2. Increased to 3 layer GAT: 29.12%
3. Upgrade to GATv2Conv with 3 layers + 3 linear prediction layers. 24.92% -> 30.36% (stopped early.)
4. Expand the `node_features` information for each node. Tanked the starting accuracy to 16%
5. Added node feature encoding layer + fixed normalization. 24.44%
6. Increase hidden layers to 64. 24.26 -> 35.55%! Sudden jump around epoch 55. Progress!
7. Add Xavier Initialization -> 37.83% vs. 34.86% with He
8. Adjust gain/bias with slight initial values. 33.15?
9. Remove gain/bias, dropout -> 0.15 45.19% huge jump! dropout -> 0.15
10. Thought I introduced residual connections, but they never applied!
11. Use the takeaway from paper ADGAN to use Initial Residual Connections + Tuneable Beta
12. Introduced Automatic Mixed Precision for speedups -> 41.77%
13. Increased the batch size to 32 and hidden channels -> 128. 60%! Huge.
14. Increased hidden channels to 256. Overfitting issues. Peak 64%
15. Increase dropout and increase L2 regularization -> weight_decay 0.05. 63%
16. Increased number of layers to 4. 64.5%
17. Increasd number of layers to 5. 63%
18. Refactored so upstream nodes are actually contextually relevant. Most likely big decrease in accuracy. 43%


# Notes
- [x] Look to improve the quality of the training examples. Filter out stubs, etc.
- [x] Add edge_attr information.
- [x] Standardize common functions for saving model checkpoints.
- [x] Python Panel Implementation
- [x] Train from the Python Panel?
- [x] Easy way to kick off training?
- [x] Ensure async requests.
- [x] Wire up final UI elements - Selected Node, Fix Selection Dialog, etc. 
- [x] Implement refresh.

# setup notes
- [ ] Automate for user.