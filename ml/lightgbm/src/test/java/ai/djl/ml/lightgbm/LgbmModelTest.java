/*
 * Copyright 2022 Amazon.com, Inc. or its affiliates. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License"). You may not use this file except in compliance
 * with the License. A copy of the License is located at
 *
 * http://aws.amazon.com/apache2.0/
 *
 * or in the "license" file accompanying this file. This file is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES
 * OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions
 * and limitations under the License.
 */

package ai.djl.ml.lightgbm;

import ai.djl.ModelException;
import ai.djl.inference.Predictor;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ModelZoo;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.training.util.DownloadUtils;
import ai.djl.translate.TranslateException;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.FloatBuffer;
import java.nio.file.Path;
import java.nio.file.Paths;
import org.testng.Assert;
import org.testng.SkipException;
import org.testng.annotations.Test;

public class LgbmModelTest {

    @Test
    public void testLoad() throws ModelException, IOException, TranslateException {
        if ("aarch64".equals(System.getProperty("os.arch"))) {
            throw new SkipException("This test requires a non-arm os.");
        }
        Path modelDir = Paths.get("build/model");
        DownloadUtils.download(
                "https://resources.djl.ai/test-models/lightgbm/quadratic.txt",
                modelDir.resolve("quadratic.txt").toString());

        Criteria<NDList, NDList> criteria =
                Criteria.builder()
                        .setTypes(NDList.class, NDList.class)
                        .optModelPath(modelDir)
                        .optModelName("quadratic")
                        .build();

        try (ZooModel<NDList, NDList> model = ModelZoo.loadModel(criteria);
                Predictor<NDList, NDList> predictor = model.newPredictor()) {
            try (NDManager manager = NDManager.newBaseManager()) {
                ByteBuffer data = ByteBuffer.allocate(10 * 4 * 4);
                FloatBuffer floats = data.asFloatBuffer();
                for (int i = 0; i < 10 * 4; i++) {
                    floats.put(1);
                }
                floats.rewind();
                NDArray array = manager.create(data, new Shape(10, 4), DataType.FLOAT32);
                NDList output = predictor.predict(new NDList(array));
                Assert.assertEquals(output.singletonOrThrow().getDataType(), DataType.FLOAT32);
                Assert.assertEquals(output.singletonOrThrow().getShape().size(), 10);
            }
        }
    }
}
