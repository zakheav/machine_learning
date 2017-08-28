package util;

/**
 * Created by yuanxi.wy on 2017/8/25.
 */
public class Vector {
    public static double[] add(double[] X, double[] Y) {
        double[] result = new double[X.length];
        for (int i = 0; i < X.length; ++i) {
            result[i] = X[i] + Y[i];
        }
        return result;
    }

    public static double dotProduct(double[] X, double[] Y) {// 点乘
        double r = 0.0;
        for (int i = 0; i < X.length; ++i) {
            r += X[i] * Y[i];
        }
        return r;
    }

    public static double[] scalarMulti(double a, double[] X) {// 标量乘
        double[] result = new double[X.length];
        for (int i = 0; i < X.length; ++i) {
            result[i] = X[i] * a;
        }
        return result;
    }

    public static double mold(double[] X) {
        return Math.sqrt(dotProduct(X, X));
    }

//    public static double[] copy(double[] X) {
//        double[] result = new double[X.length];
//        for(int i = 0; i < result.length; ++i) {
//            result[i] = X[i];
//        }
//        return result;
//    }
}
