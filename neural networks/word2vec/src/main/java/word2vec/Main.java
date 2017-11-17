package word2vec;

import util.GenerateDict;
import util.HuffumanTree;
import util.Vector;
import util.Word;

import java.text.DecimalFormat;
import java.util.*;

/**
 * Created by yuanxi.wy on 2017/8/25.
 */
class MyComparator implements Comparator<Word> {
    public int compare(Word w1, Word w2) {
        int i1 = Integer.parseInt(w1.word);
        int i2 = Integer.parseInt(w2.word);
        return i1 - i2;
    }
}
public class Main {
    public static void analysis(List<Word> dict) {// 获取词向量的相似度矩阵
        for (int i = 0; i < dict.size(); ++i) {
            System.out.println();
            System.out.print(dict.get(i).word + " ");
            for (int j = 0; j < dict.size(); ++j) {
                double dis = Vector.mold(Vector.add(dict.get(i).vec, Vector.scalarMulti(-1.0, dict.get(j).vec)));
                double cos = Vector.dotProduct(dict.get(i).vec, dict.get(j).vec) / (Vector.mold(dict.get(i).vec) * Vector.mold(dict.get(j).vec));
                DecimalFormat df = new DecimalFormat("0.#####");
                System.out.print(df.format(dis) + ", ");
            }
        }
        System.out.println();
    }

    public static void main(String[] args) {
        List<Word> dict = GenerateDict.getDict();// 构建词典
        HuffumanTree huffumanTree = new HuffumanTree(dict);// 根据词典构建huffuman树

        for (int i = 0; i < dict.size(); ++i) {
            System.out.print(huffumanTree.treeArr[i].word + ": ");
            for (int c : huffumanTree.treeArr[i].code) {
                System.out.print(c + ",");
            }
            System.out.println();
        }

        Map<String, Integer> word_idx = new HashMap<String, Integer>();// 单词在huffuman树中的下标
        for (int i = 0; i < dict.size(); ++i) {
            word_idx.put(huffumanTree.treeArr[i].word, i);
        }

        Train.train(huffumanTree, word_idx);// 训练

        // display
        List<Word> result = new ArrayList<Word>();

        // display
        for (Word word : dict) {
            result.add(word);
        }
        result.sort(new MyComparator());
        for (Word word : result) {
            System.out.println(word.toString());
        }

        // analysis
        analysis(result);
    }
}
