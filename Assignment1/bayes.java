import java.io.*;
import java.util.*;
public class bayes{
    public static void main(String[] args) {
        if(args.length < 4){
            System.out.println("Please specify the correct input files.");
            System.exit(1);
        }
        File trainData = new File(args[0]);
        File trainLabels = new File(args[1]);

        File testData = new File(args[2]);
        File testLabels = new File(args[3]);

        int[] t_labels = getLabels(trainLabels);
        ArrayList<ArrayList<String>> docs = getDocs(trainData);
        //Figure out a better way to get the number of classes
        int[] classes = {0,1};
        ArrayList<String> vocab = getVocab(docs);
        double[] prior = new double[classes.length];
        double[][] condprob = new double[vocab.size()][classes.length];

        train_bernoulli_nb(classes, docs, trainLabels, vocab, prior, condprob, t_labels);

        int[] test1_labels = apply_bernoulli_nb(classes, vocab, prior, condprob, trainData);
        //int[] real_test1_labels = getLabels(trainLabels);
        double accuracy1 = score(t_labels, test1_labels);

        int[] test2_labels = apply_bernoulli_nb(classes, vocab, prior, condprob, testData);
        int[] real_test2_labels = getLabels(testLabels);
        double accuracy2 = score(real_test2_labels, test2_labels);
        
        String t1 = "Test 1 using " + args[0] + " and " + args[1] + " to train and test.\nAccuracy: " + accuracy1*100 + "%\n";
        String t2 = "Test 2 using " + args[0] + " and " + args[1] + " to train and " + args[2] + " and " + args[3] + " to test.\nAccuracy: " + accuracy2*100 + "%\n";

        BufferedWriter output = null;
        try {
            File results = new File("results.txt");
            output = new BufferedWriter(new FileWriter(results));
            output.write(t1);
            output.write("\n");
            output.write(t2);
            if ( output != null ) {
                output.close();
              }
        } catch ( IOException e ) {
            e.printStackTrace();
        } 
    }

    public static void train_bernoulli_nb(int[] classes, ArrayList<ArrayList<String>> docs, File f, ArrayList<String> vocab, double[] prior, double[][] condprob, int[] labels) {
        int n = docs.size();
        for(int c : classes){
            int Nc = CountDocsInClass(c, f);
            prior[c] = (double)Nc/(double)n;
            for(String term : vocab){
                int index = vocab.indexOf(term);
                int Nct = CountDocsInClassContainingTerm(docs, c, term, labels);
                //System.out.println(term + " " + Nct);
                condprob[index][c] = (double)(Nct+1)/(double)(Nc+2);
            }
        }
    }

    public static int[] apply_bernoulli_nb(int[] classes, ArrayList<String> vocab, double[] prior, double[][] condprob, File testData) {
        ArrayList<ArrayList<String>> docs = getDocs(testData); 
        int[] labels = new int[docs.size()];
        for(int i = 0; i < docs.size(); i++){
            ArrayList<String> Vd = getVocabDoc(docs.get(i));
            double[] score = new double[classes.length];
            for(int c : classes){
                score[c] = Math.log(prior[c]);
                for(String term : vocab){
                    int index = vocab.indexOf(term);
                    if(Vd.contains(term)){
                        score[c] += Math.log(condprob[index][c]);
                    } else {
                        score[c] += Math.log(1-condprob[index][c]);
                    }
                }
            }
            labels[i] = max_score(score);
        }
        return labels;
    }

    public static double score(int[] labels, int[] real_labels) {
        int correct = 0;
        for(int i = 0; i < labels.length; i++){
            if(labels[i] == real_labels[i])
                correct++;
        }
        return (double)correct/(double)labels.length;
    }

    public static int[] getLabels(File f) {
        ArrayList<Integer> a = new ArrayList<Integer>();
        try {
            Scanner s = new Scanner(f);

            while (s.hasNextInt()) {
                a.add(s.nextInt());
            }
            s.close();
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        }
        int[] b = new int[a.size()];
        for(int i = 0; i < b.length; i++){
            b[i] = a.get(i);
        }
        return b;
    }

    public static ArrayList<ArrayList<String>> getDocs(File f) {
        ArrayList<String> docs = new ArrayList<String>();
        try {
            Scanner s = new Scanner(f);

            while (s.hasNextLine()) {
                String line = s.nextLine();
                docs.add(line);
            }
            s.close();
        } 
        catch (FileNotFoundException e) {
            e.printStackTrace();
        }
        ArrayList<ArrayList<String>> d = new ArrayList<ArrayList<String>>();
        for(int i = 0; i < docs.size(); i++){
            StringTokenizer st = new StringTokenizer(docs.get(i));
            ArrayList<String> line = new ArrayList<String>();
            while (st.hasMoreTokens()) {
                line.add(st.nextToken());
            }
            d.add(line);
        }
        return d;
    }

    public static ArrayList<String> getVocab(ArrayList<ArrayList<String>> docs){
        ArrayList<String> vocab = new ArrayList<String>();
        for(int i = 0; i < docs.size(); i++){
            for(int j = 0; j < docs.get(i).size(); j++){
                if(!vocab.contains(docs.get(i).get(j)))
                    vocab.add(docs.get(i).get(j));
            }
        }
        return vocab;
    }

    public static int CountDocsInClass(int c, File f) {
        int count = 0;
        try {
            Scanner s = new Scanner(f);

            while (s.hasNextInt()) {
                int n = s.nextInt();
                
                if(n == c)
                    count++;
            }
            s.close();
        } 
        catch (FileNotFoundException e) {
            e.printStackTrace();
        }
        return count;
    }

    public static int CountDocsInClassContainingTerm(ArrayList<ArrayList<String>> docs, int c, String term, int[] labels) {
        int count = 0;
        for(ArrayList<String> doc : docs){
            if(doc.contains(term) && labels[docs.indexOf(doc)] == c)
                count++;
        }
        return count;
    }

    public static ArrayList<String> getVocabDoc(ArrayList<String> docs){
        ArrayList<String> vocab = new ArrayList<String>();
        for(int i = 0; i < docs.size(); i++){
            if(!vocab.contains(docs.get(i)))
                vocab.add(docs.get(i));
        }
        return vocab;
    }
    public static int max_score(double[] a) {
        if(a[0] > a[1])
            return 0;
        else
            return 1;
    }
}