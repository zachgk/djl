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

import ai.djl.Device;
import ai.djl.engine.Engine;
import ai.djl.ndarray.BaseNDManager;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import java.nio.Buffer;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Path;

/** {@code LgbmNDManager} is the LightGBM implementation of {@link NDManager}. */
public class LgbmNDManager extends BaseNDManager {

    private static final LgbmNDManager SYSTEM_MANAGER = new SystemManager();

    private LgbmNDManager(NDManager parent, Device device) {
        super(parent, device);
    }

    static LgbmNDManager getSystemManager() {
        return SYSTEM_MANAGER;
    }

    /** {@inheritDoc} */
    @Override
    public ByteBuffer allocateDirect(int capacity) {
        return ByteBuffer.allocateDirect(capacity).order(ByteOrder.nativeOrder());
    }

    /**
     * Converts from another engines {@link NDArray}.
     *
     * @param array an array from any engine of any type (including an {@link LgbmNDArray})
     * @return an {@link LgbmNDArray}
     */
    public LgbmNDArray from(NDArray array) {
        if (array == null || array instanceof LgbmNDArray) {
            return (LgbmNDArray) array;
        }
        return (LgbmNDArray) create(array.toByteBuffer(), array.getShape(), array.getDataType());
    }

    /** {@inheritDoc} */
    @Override
    public NDManager newSubManager(Device device) {
        LgbmNDManager manager = new LgbmNDManager(this, device);
        attachInternal(manager.uid, manager);
        return manager;
    }

    /** {@inheritDoc} */
    @Override
    public Engine getEngine() {
        return Engine.getEngine(LgbmEngine.ENGINE_NAME);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray create(Buffer data, Shape shape, DataType dataType) {
        if (data instanceof ByteBuffer) {
            // output only NDArray
            return new LgbmNDArray(this, (ByteBuffer) data, shape, dataType);
        }
        throw new UnsupportedOperationException("LgbmNDArray only supports float32.");
    }

    /** {@inheritDoc} */
    @Override
    public NDList load(Path path) {
        return new NDList(new LgbmDataset(this, path));
    }

    /** The SystemManager is the root {@link LgbmNDManager} of which all others are children. */
    private static final class SystemManager extends LgbmNDManager {

        SystemManager() {
            super(null, null);
        }

        /** {@inheritDoc} */
        @Override
        public void attachInternal(String resourceId, AutoCloseable resource) {}

        /** {@inheritDoc} */
        @Override
        public void detachInternal(String resourceId) {}

        /** {@inheritDoc} */
        @Override
        public void close() {}
    }
}
