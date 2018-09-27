import java.io.*;
import java.util.*;
public class bayes{
    public static void main(String[] args) {
        if(args.length < 5){
            System.out.println("Please specify the correct input:");
            System.out.println("java bayes trainingData.txt trainingLabels.txt testingData.txt testingLabels.txt numberOfClasses");
            System.exit(1);
        }
        File trainData = new File(args[0]);
        File trainLabels = new File(args[1]);

        File testData = new File(args[2]);
        File testLabels = new File(args[3]);

        int[] t_labels = getLabels(trainLabels);
        ArrayList<ArrayList<String>> docs = getDocs(trainData);
        //Figure out a better way to get the number of classes
        int[] classes = new int[Integer.parseInt(args[4])];
        for(int i = 0; i < classes.length; i++){
            classes[i] = i;
        }
        ArrayList<String> vocab = getVocab(docs);
        double[] prior = new double[classes.length];
        double[][] condprob = new double[vocab.size()][classes.length];

        train_bernoulli_nb(classes, docs, trainLabels, vocab, prior, condprob, t_labels);

        int[] test1_labels = apply_bernoulli_nb(classes, vocab, prior, condprob, trainData);
        //int[] real_test1_labels = getLabels(trainLabels);
        double accuracy1_1 = score(t_labels, test1_labels);

        int[] test2_labels = apply_bernoulli_nb(classes, vocab, prior, condprob, testData);
        int[] real_test2_labels = getLabels(testLabels);
        double accuracy1_2 = score(real_test2_labels, test2_labels);
        String t1 = "Testing using Bernoulli Model:";
        String t1_1 = "Test 1 using " + args[0] + " and " + args[1] + " to train and test.\nAccuracy: " + String.format("%1$.2f", accuracy1_1*100) + "%";
        String t1_2 = "Test 2 using " + args[0] + " and " + args[1] + " to train and " + args[2] + " and " + args[3] + " to test.\nAccuracy: " + String.format("%1$.2f", accuracy1_2*100) + "%";

        train_multinomial_nb(classes, docs, trainLabels, vocab, prior, condprob, t_labels);

        int[] test2_1_labels = apply_multinomial_nb(classes, vocab, prior, condprob, trainData);
        double accuracy2_1 = score(t_labels, test2_1_labels);
        int[] test2_2_labels = apply_multinomial_nb(classes, vocab, prior, condprob, testData);
        double accuracy2_2 = score(real_test2_labels, test2_2_labels);

        String t2 = "Testing using Multinomial Model:";
        String t2_1 = "Test 1 using " + args[0] + " and " + args[1] + " to train and test.\nAccuracy: " + String.format("%1$.2f", accuracy2_1*100) + "%";
        String t2_2 = "Test 2 using " + args[0] + " and " + args[1] + " to train and " + args[2] + " and " + args[3] + " to test.\nAccuracy: " + String.format("%1$.2f", accuracy2_2*100) + "%";


        BufferedWriter output = null;
        try {
            File results = new File("results.txt");
            output = new BufferedWriter(new FileWriter(results));
            output.write(t1);
            output.write("\n");
            output.write("\n");
            output.write(t1_1);
            output.write("\n");
            output.write(t1_2);
            output.write("\n");
            output.write("\n");

            output.write(t2);
            output.write("\n");
            output.write("\n");
            output.write(t2_1);
            output.write("\n");
            output.write(t2_2);
            if ( output != null ) {
                output.close();
              }
        } catch ( IOException e ) {
            e.printStackTrace();
        } 
    }

    public static void train_multinomial_nb(int[] classes, ArrayList<ArrayList<String>> docs, File f, ArrayList<String> vocab, double[] prior, double[][] condprob, int[] labels) {
        int n = docs.size();
        for(int c : classes){
            int Nc = CountDocsInClass(c, f);
            prior[c] = (double)Nc/(double)n;
            ArrayList<ArrayList<String>> textc = concatenate_text_of_docs_in_class(docs, c, labels);
            int[] Tct = new int[vocab.size()];
            for(String term : vocab){
                int index = vocab.indexOf(term);
                Tct[index] = count_tokens_of_term(textc, term);
            }
            int text_sum = sum_arr(Tct);
            for(String term : vocab){
                int index = vocab.indexOf(term);
                // int textc_length = len_tc(textc);
                condprob[index][c] = (double)(Tct[index]+1)/(double)(text_sum + vocab.size());
            }
        }
    }

    public static int sum_arr(int[] a) {
        int sum = 0;
        for(int b : a){
            sum += b;
        }
        return sum;
    }

    public static int[] apply_multinomial_nb(int[] classes, ArrayList<String> vocab, double[] prior, double[][] condprob, File testData) {
        ArrayList<ArrayList<String>> docs = getDocs(testData); 
        int[] labels = new int[docs.size()];
        for(int i = 0; i < docs.size(); i++){
            ArrayList<String> Vd = getVocabDoc(docs.get(i));
            double[] score = new double[classes.length];
            for(int c : classes){
                score[c] = Math.log(prior[c]);
                for(String term : vocab){
                    int index = vocab.indexOf(term);
                    if(Vd.contains(term))
                        score[c] += Math.log(condprob[index][c]);
                }
            }
            labels[i] = max_score(score);
        }
        return labels;
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

    public static ArrayList<ArrayList<String>> concatenate_text_of_docs_in_class(ArrayList<ArrayList<String>> docs, int c, int[] labels) {
        ArrayList<ArrayList<String>> docs_of_class_c = new ArrayList<ArrayList<String>>();
        for(ArrayList<String> doc : docs){
            if(labels[docs.indexOf(doc)] == c){
                docs_of_class_c.add(doc);
            }
        }
        return docs_of_class_c;
    }

    public static int count_tokens_of_term(ArrayList<ArrayList<String>>textc, String term) {
        int count = 0;
        for(ArrayList<String> doc : textc){
            if(doc.contains(term))
                count++;
        }
        return count;
    }

    public static int len_tc(ArrayList<ArrayList<String>> textc) {
        int len = 0;
        for(ArrayList<String> doc : textc){
            len += doc.size();
        }
        return len;
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
        int index = 0;
        double max = a[0];
        for(int i = 1; i < a.length; i++){
            if(a[i] > max){
                max = a[i];
                index = i;
            }
        }
        return index;
    }

    /*  train_multinomial_nb(C, D)
            V => extract_vocabulary(D)
            N => count_docs(D)
            for each class in C
                Nc => count_docs_in_class(D,class)
                prior[c] => Nc/N
                textc => concatenate_text_of_docs_in_class(D, class)
                for each term in V
                    Tct => count_tokens_of_term(textc, term)
                for each term in V
                    condprob[term][c] => Tct+1/Sumt'(Tct'+1)
            return V, prior, condprob
    */

    /*  apply_multinomial_nb(C, V, prior, condprob, d){
            W => extract_tokens_from_doc(V, d)
            for each class c in C
                score[c] => log prior[c]
                for each term in W
                    score[c] += log condprob[term][c]
            return arg max score[c]
    }
    */
}