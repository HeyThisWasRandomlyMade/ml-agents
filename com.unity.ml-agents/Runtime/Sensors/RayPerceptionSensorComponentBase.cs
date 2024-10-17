using System;
using System.Collections.Generic;
using Unity.Collections;
using Unity.Jobs;
using UnityEngine;

namespace Unity.MLAgents.Sensors
{
    /// <summary>
    /// Determines which dimensions the sensor will perform the casts in.
    /// </summary>
    public enum RayPerceptionCastType
    {
        /// <summary>
        /// Cast in 2 dimensions, using Physics2D.CircleCast or Physics2D.RayCast.
        /// </summary>
        Cast2D,

        /// <summary>
        /// Cast in 3 dimensions, using Physics.SphereCast or Physics.RayCast.
        /// </summary>
        Cast3D,
    }

    /// <summary>
    /// Contains the elements that define a ray perception sensor.
    /// </summary>
    public struct RayPerceptionInput
    {
        /// <summary>
        /// Length of the rays to cast. This will be scaled up or down based on the scale of the transform.
        /// </summary>
        public float RayLength;

        /// <summary>
        /// List of tags which correspond to object types agent can see.
        /// </summary>
        public IReadOnlyList<string> DetectableTags;

        /// <summary>
        /// List of angles (in degrees) used to define the rays.
        /// 90 degrees is considered "forward" relative to the game object.
        /// </summary>
        public IReadOnlyList<float> Angles;

        /// <summary>
        /// Starting height offset of ray from center of agent
        /// </summary>
        public float StartOffset;

        /// <summary>
        /// Ending height offset of ray from center of agent.
        /// </summary>
        public float EndOffset;

        /// <summary>
        /// Radius of the sphere to use for spherecasting.
        /// If 0 or less, rays are used instead - this may be faster, especially for complex environments.
        /// </summary>
        public float CastRadius;

        /// <summary>
        /// Transform of the GameObject.
        /// </summary>
        public Transform Transform;

        /// <summary>
        /// Whether to perform the casts in 2D or 3D.
        /// </summary>
        public RayPerceptionCastType CastType;

        /// <summary>
        /// Filtering options for the casts.
        /// </summary>
        public int LayerMask;

        /// <summary>
        ///  Whether to use batched raycasts.
        /// </summary>
        public bool UseBatchedRaycasts;

        /// <summary>
        /// Returns the expected number of floats in the output.
        /// </summary>
        /// <returns>The expected number of floats in the output.</returns>
        public int OutputSize()
        {
            // Modified output size: 2 observations per ray
            return 2 * (Angles?.Count ?? 0);
        }

        /// <summary>
        /// Get the cast start and end points for the given ray index/
        /// </summary>
        /// <param name="rayIndex">Ray index</param>
        /// <returns>A tuple of the start and end positions in world space.</returns>
        public (Vector3 StartPositionWorld, Vector3 EndPositionWorld) RayExtents(int rayIndex)
        {
            var angle = Angles[rayIndex];
            Vector3 startPositionLocal, endPositionLocal;
            if (CastType == RayPerceptionCastType.Cast3D)
            {
                startPositionLocal = new Vector3(0, StartOffset, 0);
                endPositionLocal = PolarToCartesian3D(RayLength, angle);
                endPositionLocal.y += EndOffset;
            }
            else
            {
                // Vector2s here get converted to Vector3s (and back to Vector2s for casting)
                startPositionLocal = new Vector2();
                endPositionLocal = PolarToCartesian2D(RayLength, angle);
            }

            var startPositionWorld = Transform.TransformPoint(startPositionLocal);
            var endPositionWorld = Transform.TransformPoint(endPositionLocal);

            return (StartPositionWorld: startPositionWorld, EndPositionWorld: endPositionWorld);
        }

        /// <summary>
        /// Converts polar coordinate to cartesian coordinate.
        /// </summary>
        static internal Vector3 PolarToCartesian3D(float radius, float angleDegrees)
        {
            var x = radius * Mathf.Cos(Mathf.Deg2Rad * angleDegrees);
            var z = radius * Mathf.Sin(Mathf.Deg2Rad * angleDegrees);
            return new Vector3(x, 0f, z);
        }

        /// <summary>
        /// Converts polar coordinate to cartesian coordinate.
        /// </summary>
        static internal Vector2 PolarToCartesian2D(float radius, float angleDegrees)
        {
            var x = radius * Mathf.Cos(Mathf.Deg2Rad * angleDegrees);
            var y = radius * Mathf.Sin(Mathf.Deg2Rad * angleDegrees);
            return new Vector2(x, y);
        }
    }

    /// <summary>
    /// Contains the data generated/produced from a ray perception sensor.
    /// </summary>
    public class RayPerceptionOutput
    {
        /// <summary>
        /// Contains the data generated from a single ray of a ray perception sensor.
        /// </summary>
        public struct RayOutput
        {
            /// <summary>
            /// Whether or not the ray hit anything.
            /// </summary>
            public bool HasHit;

            /// <summary>
            /// The index of the hit object's tag in the DetectableTags list, or -1 if there was no hit, or the
            /// hit object has a different tag.
            /// </summary>
            public int HitTagIndex;

            /// <summary>
            /// Normalized distance to the hit object.
            /// </summary>
            public float HitFraction;

            /// <summary>
            /// The hit GameObject (or null if there was no hit).
            /// </summary>
            public GameObject HitGameObject;

            /// <summary>
            /// Start position of the ray in world space.
            /// </summary>
            public Vector3 StartPositionWorld;

            /// <summary>
            /// End position of the ray in world space.
            /// </summary>
            public Vector3 EndPositionWorld;

            /// <summary>
            /// The scaled length of the ray.
            /// </summary>
            /// <remarks>
            /// If there is non-(1,1,1) scale, |EndPositionWorld - StartPositionWorld| will be different from
            /// the input rayLength.
            /// </remarks>
            public float ScaledRayLength
            {
                get
                {
                    var rayDirection = EndPositionWorld - StartPositionWorld;
                    return rayDirection.magnitude;
                }
            }

            /// <summary>
            /// The scaled size of the cast.
            /// </summary>
            /// <remarks>
            /// If there is non-(1,1,1) scale, the cast radius will be also be scaled.
            /// </remarks>
            public float ScaledCastRadius;

            /// <summary>
            /// Writes the ray output information to a subset of the float array.
            /// </summary>
            /// <param name="numTags">Number of detectable tags</param>
            /// <param name="buffer">Output buffer</param>
            /// <param name="bufferIndex">Start index in the buffer</param>
            public void ToFloatArray(int numTags, float[] buffer, int bufferIndex)
            {
                // Observation 1: Normalized distance (-1 if missed)
                buffer[bufferIndex] = HasHit ? HitFraction : -1f;

                // Observation 2: Tag value between 0 and 1 (0 if no tag or unknown tag)
                if (HasHit && HitTagIndex >= 0 && numTags > 0)
                {
                    buffer[bufferIndex + 1] = (HitTagIndex + 1) / (float)numTags;
                }
                else
                {
                    buffer[bufferIndex + 1] = 0f; // No tag or unknown tag
                }
            }
        }

        /// <summary>
        /// RayOutput for each ray that was cast.
        /// </summary>
        public RayOutput[] RayOutputs;
    }

    /// <summary>
    /// A sensor implementation that supports ray cast-based observations.
    /// </summary>
    public class RayPerceptionSensor : ISensor, IBuiltInSensor
    {
        float[] m_Observations;
        ObservationSpec m_ObservationSpec;
        string m_Name;

        RayPerceptionInput m_RayPerceptionInput;
        RayPerceptionOutput m_RayPerceptionOutput;

        bool m_UseBatchedRaycasts;

        /// <summary>
        /// Time.frameCount at the last time Update() was called. This is only used for display in gizmos.
        /// </summary>
        int m_DebugLastFrameCount;

        internal int DebugLastFrameCount
        {
            get { return m_DebugLastFrameCount; }
        }

        /// <summary>
        /// Creates the RayPerceptionSensor.
        /// </summary>
        /// <param name="name">The name of the sensor.</param>
        /// <param name="rayInput">The inputs for the sensor.</param>
        public RayPerceptionSensor(string name, RayPerceptionInput rayInput)
        {
            m_Name = name;
            m_RayPerceptionInput = rayInput;
            m_UseBatchedRaycasts = rayInput.UseBatchedRaycasts;

            SetNumObservations(rayInput.OutputSize());

            m_DebugLastFrameCount = Time.frameCount;
            m_RayPerceptionOutput = new RayPerceptionOutput();
        }

        /// <summary>
        /// The most recent raycast results.
        /// </summary>
        public RayPerceptionOutput RayPerceptionOutput
        {
            get { return m_RayPerceptionOutput; }
        }

        void SetNumObservations(int numObservations)
        {
            m_ObservationSpec = ObservationSpec.Vector(numObservations);
            m_Observations = new float[numObservations];
        }

        internal void SetRayPerceptionInput(RayPerceptionInput rayInput)
        {
            if (m_RayPerceptionInput.OutputSize() != rayInput.OutputSize())
            {
                Debug.Log(
                    "Changing the number of tags or rays at runtime is not " +
                    "supported and may cause errors in training or inference."
                );
                SetNumObservations(rayInput.OutputSize());
            }
            m_RayPerceptionInput = rayInput;
        }

        /// <summary>
        /// Computes the ray perception observations and saves them to the provided
        /// <see cref="ObservationWriter"/>.
        /// </summary>
        /// <param name="writer">Where the ray perception observations are written to.</param>
        /// <returns>The number of written observations.</returns>
        public int Write(ObservationWriter writer)
        {
            Array.Clear(m_Observations, 0, m_Observations.Length);
            int numRays = m_RayPerceptionInput.Angles.Count;
            int numTags = m_RayPerceptionInput.DetectableTags?.Count ?? 0;

            for (int rayIndex = 0; rayIndex < numRays; rayIndex++)
            {
                int bufferIndex = rayIndex * 2;
                m_RayPerceptionOutput.RayOutputs[rayIndex].ToFloatArray(numTags, m_Observations, bufferIndex);
            }

            writer.AddList(m_Observations);
            return m_Observations.Length;
        }

        /// <inheritdoc/>
        public void Update()
        {
            m_DebugLastFrameCount = Time.frameCount;
            var numRays = m_RayPerceptionInput.Angles.Count;

            if (m_RayPerceptionOutput.RayOutputs == null || m_RayPerceptionOutput.RayOutputs.Length != numRays)
            {
                m_RayPerceptionOutput.RayOutputs = new RayPerceptionOutput.RayOutput[numRays];
            }

            if (m_UseBatchedRaycasts && m_RayPerceptionInput.CastType == RayPerceptionCastType.Cast3D)
            {
                PerceiveBatchedRays(ref m_RayPerceptionOutput.RayOutputs, m_RayPerceptionInput);
            }
            else
            {
                // For each ray, do the casting and save the results.
                for (var rayIndex = 0; rayIndex < numRays; rayIndex++)
                {
                    m_RayPerceptionOutput.RayOutputs[rayIndex] = PerceiveSingleRay(m_RayPerceptionInput, rayIndex);
                }
            }
        }

        /// <inheritdoc/>
        public void Reset() { }

        /// <inheritdoc/>
        public ObservationSpec GetObservationSpec()
        {
            return m_ObservationSpec;
        }

        /// <inheritdoc/>
        public string GetName()
        {
            return m_Name;
        }

        /// <inheritdoc/>
        public virtual byte[] GetCompressedObservation()
        {
            return null;
        }

        /// <inheritdoc/>
        public CompressionSpec GetCompressionSpec()
        {
            return CompressionSpec.Default();
        }

        /// <inheritdoc/>
        public BuiltInSensorType GetBuiltInSensorType()
        {
            return BuiltInSensorType.RayPerceptionSensor;
        }

        /// <summary>
        /// Evaluates the raycasts to be used as part of an observation of an agent.
        /// </summary>
        /// <param name="input">Input defining the rays that will be cast.</param>
        /// <param name="batched">Use batched raycasts.</param>
        /// <returns>Output struct containing the raycast results.</returns>
        public static RayPerceptionOutput Perceive(RayPerceptionInput input, bool batched)
        {
            RayPerceptionOutput output = new RayPerceptionOutput();
            output.RayOutputs = new RayPerceptionOutput.RayOutput[input.Angles.Count];

            if (batched)
            {
                PerceiveBatchedRays(ref output.RayOutputs, input);
            }
            else
            {
                for (var rayIndex = 0; rayIndex < input.Angles.Count; rayIndex++)
                {
                    output.RayOutputs[rayIndex] = PerceiveSingleRay(input, rayIndex);
                }
            }

            return output;
        }

        /// <summary>
        /// Evaluate the raycast results of all the rays from the RayPerceptionInput as a batch.
        /// </summary>
        /// <param name="input">Input</param>
        internal static void PerceiveBatchedRays(ref RayPerceptionOutput.RayOutput[] batchedRaycastOutputs, RayPerceptionInput input)
        {
            var numRays = input.Angles.Count;
            var results = new NativeArray<RaycastHit>(numRays, Allocator.TempJob);
            var unscaledRayLength = input.RayLength;
            var unscaledCastRadius = input.CastRadius;

            var raycastCommands = new NativeArray<RaycastCommand>(unscaledCastRadius <= 0f ? numRays : 0, Allocator.TempJob);
            var spherecastCommands = new NativeArray<SpherecastCommand>(unscaledCastRadius > 0f ? numRays : 0, Allocator.TempJob);

            for (int i = 0; i < numRays; i++)
            {
                var extents = input.RayExtents(i);
                var startPositionWorld = extents.StartPositionWorld;
                var endPositionWorld = extents.EndPositionWorld;

                var rayDirection = endPositionWorld - startPositionWorld;
                var scaledRayLength = rayDirection.magnitude;
                var scaledCastRadius = unscaledRayLength > 0 ?
                    unscaledCastRadius * scaledRayLength / unscaledRayLength :
                    unscaledCastRadius;

                var queryParameters = QueryParameters.Default;
                queryParameters.layerMask = input.LayerMask;

                var rayDirectionNormalized = rayDirection.normalized;

                if (scaledCastRadius > 0f)
                {
                    spherecastCommands[i] = new SpherecastCommand(startPositionWorld, scaledCastRadius, rayDirectionNormalized, queryParameters, scaledRayLength);
                }
                else
                {
                    raycastCommands[i] = new RaycastCommand(startPositionWorld, rayDirectionNormalized, queryParameters, scaledRayLength);
                }

                batchedRaycastOutputs[i] = new RayPerceptionOutput.RayOutput
                {
                    HitTagIndex = -1,
                    StartPositionWorld = startPositionWorld,
                    EndPositionWorld = endPositionWorld,
                    ScaledCastRadius = scaledCastRadius
                };
            }

            if (unscaledCastRadius > 0f)
            {
                JobHandle handle = SpherecastCommand.ScheduleBatch(spherecastCommands, results, 1, 1, default(JobHandle));
                handle.Complete();
            }
            else
            {
                JobHandle handle = RaycastCommand.ScheduleBatch(raycastCommands, results, 1, 1, default(JobHandle));
                handle.Complete();
            }

            for (int i = 0; i < results.Length; i++)
            {
                var castHit = results[i].collider != null;
                var hitFraction = 1.0f;
                GameObject hitObject = null;
                float scaledRayLength;
                float scaledCastRadius = batchedRaycastOutputs[i].ScaledCastRadius;
                if (scaledCastRadius > 0f)
                {
                    scaledRayLength = spherecastCommands[i].distance;
                }
                else
                {
                    scaledRayLength = raycastCommands[i].distance;
                }

                hitFraction = castHit ? (scaledRayLength > 0 ? results[i].distance / scaledRayLength : 0.0f) : 1.0f;
                hitObject = castHit ? results[i].collider.gameObject : null;

                if (castHit && hitObject != null)
                {
                    var numTags = input.DetectableTags?.Count ?? 0;
                    for (int j = 0; j < numTags; j++)
                    {
                        try
                        {
                            var tag = input.DetectableTags[j];
                            if (!string.IsNullOrEmpty(tag) && hitObject.CompareTag(tag))
                            {
                                batchedRaycastOutputs[i].HitTagIndex = j;
                                break;
                            }
                        }
                        catch (UnityException)
                        {
                        }
                    }
                }

                batchedRaycastOutputs[i].HasHit = castHit;
                batchedRaycastOutputs[i].HitFraction = hitFraction;
                batchedRaycastOutputs[i].HitGameObject = hitObject;
            }

            results.Dispose();
            raycastCommands.Dispose();
            spherecastCommands.Dispose();
        }

        /// <summary>
        /// Evaluate the raycast results of a single ray from the RayPerceptionInput.
        /// </summary>
        /// <param name="input">Input</param>
        /// <param name="rayIndex">Ray index</param>
        /// <returns>`RayOutput` result of a single raycast.</returns>
        internal static RayPerceptionOutput.RayOutput PerceiveSingleRay(
            RayPerceptionInput input,
            int rayIndex
        )
        {
            var unscaledRayLength = input.RayLength;
            var unscaledCastRadius = input.CastRadius;

            var extents = input.RayExtents(rayIndex);
            var startPositionWorld = extents.StartPositionWorld;
            var endPositionWorld = extents.EndPositionWorld;

            var rayDirection = endPositionWorld - startPositionWorld;
            var scaledRayLength = rayDirection.magnitude;
            var scaledCastRadius = unscaledRayLength > 0 ?
                unscaledCastRadius * scaledRayLength / unscaledRayLength :
                unscaledCastRadius;

            var castHit = false;
            var hitFraction = 1.0f;
            GameObject hitObject = null;

            if (input.CastType == RayPerceptionCastType.Cast3D)
            {
#if MLA_UNITY_PHYSICS_MODULE
                RaycastHit rayHit;
                if (scaledCastRadius > 0f)
                {
                    castHit = Physics.SphereCast(startPositionWorld, scaledCastRadius, rayDirection, out rayHit,
                        scaledRayLength, input.LayerMask);
                }
                else
                {
                    castHit = Physics.Raycast(startPositionWorld, rayDirection, out rayHit,
                        scaledRayLength, input.LayerMask);
                }

                hitFraction = castHit ? (scaledRayLength > 0 ? rayHit.distance / scaledRayLength : 0.0f) : 1.0f;
                hitObject = castHit ? rayHit.collider.gameObject : null;
#endif
            }
            else
            {
#if MLA_UNITY_PHYSICS2D_MODULE
                RaycastHit2D rayHit;
                if (scaledCastRadius > 0f)
                {
                    rayHit = Physics2D.CircleCast(startPositionWorld, scaledCastRadius, rayDirection,
                        scaledRayLength, input.LayerMask);
                }
                else
                {
                    rayHit = Physics2D.Raycast(startPositionWorld, rayDirection, scaledRayLength, input.LayerMask);
                }

                castHit = rayHit;
                hitFraction = castHit ? rayHit.fraction : 1.0f;
                hitObject = castHit ? rayHit.collider.gameObject : null;
#endif
            }

            var rayOutput = new RayPerceptionOutput.RayOutput
            {
                HasHit = castHit,
                HitFraction = hitFraction,
                HitTagIndex = -1,
                HitGameObject = hitObject,
                StartPositionWorld = startPositionWorld,
                EndPositionWorld = endPositionWorld,
                ScaledCastRadius = scaledCastRadius
            };

            if (castHit && hitObject != null)
            {
                int numTags = input.DetectableTags?.Count ?? 0;
                for (int i = 0; i < numTags; i++)
                {
                    try
                    {
                        var tag = input.DetectableTags[i];
                        if (!string.IsNullOrEmpty(tag) && hitObject.CompareTag(tag))
                        {
                            rayOutput.HitTagIndex = i;
                            break;
                        }
                    }
                    catch (UnityException)
                    {
                        // Ignore invalid tags
                    }
                }
            }

            return rayOutput;
        }
    }
}
