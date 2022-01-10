def HydrogenIB_fgsm(model, epsilon, advType):
    model = load(model, "HydrogenIB")
    model.eval()
    model.train_flag = False
    if advType == "fgsm":
        adver_image_obtain = fgsm.attack_model(model=model)
    elif advType == "pgd":
        adver_image_obtain = pgd.attack_model(model=model)
    accuracy_clean = []
    accuracy_adver = []
    # fuck = 0
    for x_batch, y_batch in test_loader_causal:

        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        y_pre, z_scores, features, logits, mean_Cs, std_Cs, y_logits_s = model(x_batch)
        y_prediction = torch.max(y_pre, dim=1)[1]
        accuracy = torch.mean((y_prediction == y_batch).float())
        accuracy_clean.append(accuracy.item())


        perturbed_x_batch = adver_image_obtain.generate(x_batch, eps=epsilon, y=y_batch)

        y_pre, z_scores, features, logits, mean_Cs, std_Cs, y_logits_s = model(x_batch)
        y_prediction = torch.max(y_pre, dim=1)[1]
        accuracy = torch.mean((y_prediction == y_batch).float())
        accuracy_adver.append(accuracy.item())


    # if(epoch%5 == 0):
    print("TEST, Clean Accuracy: ", np.mean(accuracy_clean), ", Adversial Accuracy: ",  np.mean(accuracy_adver))
