package util;

import java.util.List;
import java.util.Map;

/**
 * Created by yuanxi.wy on 2017/8/21.
 */
public class Word {
    public String word;
    public long counter;
    public List<Integer> code;// 每个元素0代表正例，1代表负例，顺序自底向上
    public double[] vec;// 每个单词的词向量(跳过被预测的词)
    public double[] theta;// 非叶子结点的参数向量
    private void init() {
        vec = new double[Constants.VEC_SIZE];
        for(int i = 0; i < vec.length; ++i) {
            vec[i] = (Math.random() - 0.5) / 100;
        }
        theta = new double[Constants.VEC_SIZE];
        for(int i = 0; i < theta.length; ++i) {
            theta[i] = (Math.random() - 0.5) / 100;
        }
    }
    public Word() {
        init();
    }
    public Word(String w) {
        init();
        word = w;
    }
    public String toString() {
        StringBuilder sb = new StringBuilder();
        sb.append(word);
        sb.append(": ");
        for(int i = 0; i < vec.length; ++i) {
            sb.append(i == vec.length - 1 ? vec[i]+"" : vec[i]+", ");
        }
        return sb.toString();
    }
}
