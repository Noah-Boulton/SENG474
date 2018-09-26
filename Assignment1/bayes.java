import java.io.*;
import java.util.*;
public class bayes{
    public static void main(String[] args) {
        if(args.length < 2){
            System.out.println("Please specify the correct input files.");
            System.exit(1);
        }
        File trainData = new File(args[0]);
        File trainLabels = new File(args[1]);

        ArrayList<ArrayList<String>> docs = getDocs(trainData);
        //Figure out a better way to get the number of classes
        int[] classes = {0,1};
        ArrayList<String> vocab = getVocab(docs);
        float[] prior = new float[classes.length];
        float[][] condprob = new float[vocab.size()][classes.length];

        train_bernoulli_nb(classes, docs, trainLabels, vocab, prior, condprob);

        // System.out.println("Classes");
        // for(int c : classes){
        //     System.out.println(c);
        // }
        // System.out.println();
        // System.out.println();
        // System.out.println("Documents");
        // for(ArrayList<String> doc : docs){
        //     System.out.println(doc);
        // }
        // System.out.println();
        // System.out.println();
        // System.out.println("Vocab");
        // for(String term : vocab){
        //     System.out.println(term);
        // } 
        // System.out.println();
        // System.out.println();
        System.out.println("Prior");  
        for(float p : prior){
            System.out.println(p);
        }
        System.out.println();
        System.out.println();
        System.out.println("Condprob");
        for(float[] cp : condprob){
            for(float k : cp){
                System.out.println(k);
            }
        }
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

    public static void train_bernoulli_nb(int[] classes, ArrayList<ArrayList<String>> docs, File f, ArrayList<String> vocab, float[] prior, float[][] condprob) {
        // vocab = getVocab(docs);
        int n = docs.size();
        // prior = new float[classes.length];
        // condprob = new float[vocab.size()][classes.length];
        for(int c : classes){
            int Nc = CountDocsInClass(c, f);
            prior[c] = (float)Nc/(float)n;
            for(String term : vocab){
                int index = vocab.indexOf(term);
                int Nct = CountDocsInClassContainingTerm(docs, c, term);
                condprob[index][c] = (float)(Nct+1)/(float)(Nc +2);
            }
        }
        // return vocab, prior, condprob
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

    public static int CountDocsInClassContainingTerm(ArrayList<ArrayList<String>> docs, int c, String term) {
        int count = 0;
        for(ArrayList<String> doc : docs){
            if(doc.contains(term))
                count++;
        }
        return count;
    }
}