package util;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Created by yuanxi.wy on 2017/8/25.
 */
public class GenerateDict {
    private static String[] test= {"a,b,c","a,y,c","l,m,n"};
    private static int counter = -1;

    public static String[] getSentence() {
        ++counter;
        if (counter == Constants.TIME) {
            counter = -1;
            return null;
        } else {
            return test[counter % test.length].split(",");
        }
    }

    public static List<Word> getDict() {
        String[] sentence = getSentence();
        Map<String, Integer> wordsCounter = new HashMap<String, Integer>();
        while (sentence != null) {
            for (String word : sentence) {
                int counter = wordsCounter.containsKey(word) ? wordsCounter.get(word) + 1 : 1;
                wordsCounter.put(word, counter);
            }
            sentence = getSentence();
        }
        List<Word> dict = new ArrayList<Word>();
        for (String w : wordsCounter.keySet()) {
            Word word = new Word(w);
            word.counter = wordsCounter.get(w);
            dict.add(word);
        }
        return dict;
    }
}
