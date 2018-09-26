import java.io.*;
import java.util.*;
public class bayes{
    public static void main(String[] args) {
        File f = new File(args[0]);
        ArrayList<ArrayList<String>> docs = getDocs(f);
        ArrayList<String> vocab = getVocab(docs);
        int[] classes = {0,1};

        for(int i = 0; i < docs.size(); i++){
            System.out.println(docs.get(i));
        }
        System.out.println();
        for(int i = 0; i < vocab.size(); i++){
            System.out.println(vocab.get(i));
        }
        System.out.println();
        System.out.println();
        // String[][] strings = new String[11110][100];
        // strings = docs.toArray(strings);

        // for(int i = 0; i < strings.length; i++){
        //     for(int j = 0; j < strings[i].length; j++){
        //         System.out.println(strings[i][j]);
        //     }
        // }
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
}

/*  Train_BernoulliNB(Classes, Documents)
        V => ExtractVocavulary(Documents)
        N => CountDocs(Documents)
        for each Class in Classes
            do Nc => CountDocsInClass(Documents, class)
               prior[c] => Nc/N
               for each term in V
               do Nct => CountDocsInClassContainingTerm(Documents, c, t)
                    condprob[t][c] => (Nc+1)/(Nc +2)
        return V, prior, condprob 
*/