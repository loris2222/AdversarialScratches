
# IN EXPERIMENTS, TO COPY WITH LABELS TAKEN FROM FILE IF COMPUTED INCORRECTLY
#reader = csv.reader(open(os.path.join(basepath, "../experiments/500sample_10kiter_1bezier50long1wide_ip.csv"), 'r', newline=''), delimiter=',')
#header = next(reader)
#outputcsv.writerow(
#    ['image name', 'target label', 'original_label', 'adversarial_label', 'queries', 'ell0', '0..n params'])

#for r in range(500):
#    line = next(reader)
#    outputcsv.writerow(
#        [line[0], line[1], np.argmax(preds[r, :]), line[3], line[4], line[5]])
#outputfile.flush()
#exit(0)

