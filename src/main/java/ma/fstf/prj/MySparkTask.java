package ma.fstf.prj;

import com.google.common.base.Preconditions;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.ml.classification.LogisticRegression;
import org.apache.spark.ml.classification.LogisticRegressionModel;
import org.apache.spark.ml.feature.StringIndexer;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.ml.feature.Imputer;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.SaveMode;
import org.apache.spark.sql.functions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class MySparkTask {
    private static final Logger LOGGER = LoggerFactory.getLogger(MySparkTask.class);

    public static void main(String[] args) {
        Preconditions.checkArgument(args.length > 1, "Please provide the path of input file and output dir as parameters.");
        new MySparkTask().run(args[0], args[1]);
    }

    public void run(String inputFilePath, String outputDir) {

        System.out.println(" inputFilePath: " + inputFilePath);
       SparkConf conf = new SparkConf().setAppName("CHDPredictionTask").setMaster("local[*]");
      // SparkConf conf = new SparkConf().setAppName(MySparkTask.class.getName());

        //System.out.println(" etape: 2" );
     //   JavaSparkContext sc = new JavaSparkContext(conf);
        JavaSparkContext sc = new JavaSparkContext(conf);

        //System.out.println(" etape: 3" );

        SparkSession spark = SparkSession.builder().appName("CHDPredictionTask").getOrCreate();
        //System.out.println(" etape: 4" );

        // Charger les données
        //train et test
        Dataset<Row> trainData = spark.read().option("header", "true").csv(inputFilePath + "/train.csv");
        System.out.println(" etape: 5" );
        Dataset<Row> testData = spark.read().option("header", "true").csv(inputFilePath + "/test.csv");
        System.out.println(" etape: 6" );

        // Supprimer les colonnes "id" et "education"
        trainData = trainData.drop("id").drop("education");
        System.out.println(" etape: 7" +trainData);
        testData = testData.drop("id").drop("education");
        System.out.println(" etape: 8" +testData);

        // Traiter les données pour convertir "Sex" et "is_smoking" en valeurs numériques
        trainData = trainData.withColumn("Sex", functions.when(trainData.col("Sex").equalTo("M"), 1.0).otherwise(0.0))
                .withColumn("is_smoking", functions.when(trainData.col("is_smoking").equalTo("YES"), 1.0).otherwise(0.0))
                .na().fill(0);
        System.out.println(" etape: 9" +trainData);
        testData = testData.withColumn("Sex", functions.when(testData.col("Sex").equalTo("M"), 1.0).otherwise(0.0))
                .withColumn("is_smoking", functions.when(testData.col("is_smoking").equalTo("YES"), 1.0).otherwise(0.0))
                .na().fill(0);
        System.out.println(" etape: 10" +testData);

        // Convertir les colonnes en double si elles contiennent des valeurs numériques
        String[] numericCols = new String[]{"age", "cigsPerDay", "BPMeds", "prevalentStroke",
                "prevalentHyp", "diabetes", "totChol", "sysBP", "diaBP", "BMI", "heartRate", "glucose"};
        for (String col : numericCols) {
            trainData = trainData.withColumn(col, functions.col(col).cast("double"));
            testData = testData.withColumn(col, functions.col(col).cast("double"));
        }

        // Imputer pour remplacer les valeurs manquantes par la moyenne
        Imputer imputer = new Imputer()
                .setInputCols(new String[]{"age", "cigsPerDay", "BPMeds", "prevalentStroke",
                        "prevalentHyp", "diabetes", "totChol", "sysBP", "diaBP", "BMI", "heartRate", "glucose"})
                .setOutputCols(new String[]{"age", "cigsPerDay", "BPMeds", "prevalentStroke",
                        "prevalentHyp", "diabetes", "totChol", "sysBP", "diaBP", "BMI", "heartRate", "glucose"})
                .setStrategy("mean");

        trainData = imputer.fit(trainData).transform(trainData);
        testData = imputer.fit(testData).transform(testData);

        // Indexer la colonne cible dans les données train
        StringIndexer labelIndexer = new StringIndexer()
                .setInputCol("TenYearCHD")
                .setOutputCol("label");

        trainData = labelIndexer.fit(trainData).transform(trainData);

        // Assembler les colonnes de features
        VectorAssembler assembler = new VectorAssembler()
                .setInputCols(new String[]{"Sex", "age", "is_smoking","cigsPerDay","BPMeds","prevalentStroke","prevalentHyp","diabetes","totChol","sysBP","diaBP","BMI","heartRate","glucose"})
                .setOutputCol("features");

        trainData = assembler.transform(trainData);
        testData = assembler.transform(testData);

        //System.out.println("Train data after assembler: ");
        trainData.show();

        //System.out.println("Test data after assembler: ");
        testData.show();

        //System.out.println(" etape: 12" +assembler);

        // Créer le modèle de régression logistique
        LogisticRegression lr = new LogisticRegression()
                .setLabelCol("label")
                .setFeaturesCol("features");


        //System.out.println(" etape: 13" );

        // Diviser les données en train et test
        Dataset<Row>[] splits = trainData.randomSplit(new double[]{0.8, 0.2});
        Dataset<Row> train = splits[0];
        Dataset<Row> test = splits[1];

        // Entraîner le modèle
        LogisticRegressionModel model = lr.fit(train);



        // Faire des prédictions sur les données de test
        Dataset<Row> predictions = model.transform(test);

        // Calculer l'exactitude
        Dataset<Row> correctPredictions = predictions.filter(predictions.col("label").equalTo(predictions.col("prediction")));
        double accuracy = (double) correctPredictions.count() / (double) test.count();
        System.out.println("Accuracy: " + accuracy);


        // Faire des prédictions sur les données testData
        Dataset<Row> testDataPredictions = model.transform(testData);

        // Sélectionner les caractéristiques et les prédictions
        Dataset<Row> output = testDataPredictions.select( "age", "sex", "is_smoking", "prediction");

        // Écrire les caractéristiques et les prédictions dans un fichier CSV
        output.write().mode(SaveMode.Overwrite).csv(outputDir);

    }
}